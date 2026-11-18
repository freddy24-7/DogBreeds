"""
Stage 5: transfer learning. Keep the pre-trained "eyes", train a new "brain".

An ImageNet model has two parts:

    photo --> [ backbone ] --> 1,408 numbers --> [ head ] --> 1,000 scores
              (almost all                         (one Linear
               of the model)                       layer)

The backbone turns a photo into 1,408 numbers that describe it (fur, ears,
snout...). The head turns those numbers into one score per ImageNet class.

In stage 5 we FREEZE the backbone (we never change it) and train a NEW head
with 120 outputs, one per Stanford Dogs breed. Because the backbone never
changes, the 1,408 numbers for a photo never change either. So we compute them
once for every photo and save them ("caching features"). Training the head on
the saved numbers then takes seconds instead of hours.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------- 1. features

def backbone(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The frozen part of an EfficientNet: photo batch [N, 3, H, W] -> features [N, 1408].

    torchvision's EfficientNet.forward() is: features -> avgpool -> flatten -> classifier.
    This is the same, minus the classifier (the ImageNet head).
    """
    return torch.flatten(model.avgpool(model.features(x)), 1)


def extract_features(predictor, dataset, device: str, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Runs the backbone once on every photo. Returns (features [N, 1408], labels [N]).

    Same settings as stage 4: one process (num_workers=0) so an 8 GB Mac doesn't run out of memory.
    """
    dataset.transform = predictor.weights.transforms()   # exactly as the model was trained: resize, crop, normalise
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    model = predictor.model.to(device)
    feats, labels = [], []
    with torch.inference_mode():
        for i, (x, y) in enumerate(loader):
            feats.append(backbone(model, x.to(device)).cpu())
            labels.append(y)
            if i % 50 == 0:
                print(f"    {i * batch_size:>6} / {len(dataset)} photos", flush=True)
    predictor.model.to("cpu")
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


# ---------------------------------------------------------------- 2. the new head

class BreedHead(nn.Module):
    """Our new head: 1,408 features -> 120 breed scores.

    The same shape as EfficientNet's own head (Dropout, then one Linear layer),
    just with 120 outputs instead of 1,000. Dropout randomly zeroes 30% of the
    features during training only, so the head can't lean on a few of them.
    """

    def __init__(self, in_features: int = 1408, n_breeds: int = 120, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features, n_breeds)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(features))


def head_from_imagenet(imagenet_linear: nn.Linear, imagenet_indices: list[int], dropout: float = 0.3) -> BreedHead:
    """A new head that starts as a copy of ImageNet's head, keeping only the rows of our 120 breeds.

    Before any training it gives exactly the stage 4 predictions: a softmax over 120 of
    the 1,000 scores is the same as the 1,000 probabilities with 120 kept and rescaled.
    """
    head = BreedHead(imagenet_linear.in_features, len(imagenet_indices), dropout)
    with torch.no_grad():
        head.linear.weight.copy_(imagenet_linear.weight[imagenet_indices])
        head.linear.bias.copy_(imagenet_linear.bias[imagenet_indices])
    return head


# ---------------------------------------------------------------- 3. train / validation split

def stratified_split(labels: np.ndarray, val_share: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Splits photo numbers into (train, validation), taking the same share from every breed.

    The validation photos are used to compare settings and to decide when to stop training.
    The test photos are never used for any of that: they're only for the final report.
    """
    rng = np.random.default_rng(seed)
    train, val = [], []
    for breed in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == breed))
        n_val = max(1, round(len(idx) * val_share))
        val += idx[:n_val].tolist()
        train += idx[n_val:].tolist()
    return np.sort(train), np.sort(val)


# ---------------------------------------------------------------- 4. training loop

@dataclass
class Epoch:
    epoch: int
    train_loss: float     # average loss on the training photos during this epoch (dropout on)
    val_loss: float       # loss on the validation photos after this epoch (dropout off)
    val_accuracy: float   # top-1 accuracy on the validation photos


def evaluate(head: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """(loss, accuracy) of the head on some photos, with dropout switched off."""
    head.eval()
    with torch.inference_mode():
        logits = head(x)
        return nn.functional.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


def train_head(head: nn.Module, train_x, train_y, val_x, val_y, epochs: int = 30, lr: float = 1e-3,
               batch_size: int = 256, weight_decay: float = 1e-4, seed: int = 0):
    """The classic PyTorch training loop, on cached features.

    Returns (best_state, history, best_epoch). best_state holds the weights from the epoch
    with the lowest validation loss ("early stopping": later epochs only made it worse).
    """
    torch.manual_seed(seed)                                   # same shuffling and dropout every run
    train_x, train_y = torch.as_tensor(train_x), torch.as_tensor(train_y, dtype=torch.long)
    val_x, val_y = torch.as_tensor(val_x), torch.as_tensor(val_y, dtype=torch.long)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    val_loss, val_acc = evaluate(head, val_x, val_y)
    history = [Epoch(0, float("nan"), val_loss, val_acc)]    # epoch 0: before any training
    best_loss, best_epoch, best_state = val_loss, 0, {k: v.clone() for k, v in head.state_dict().items()}

    for epoch in range(1, epochs + 1):
        head.train()                                          # dropout on
        order = torch.randperm(len(train_x))                  # a new random order every epoch
        total = 0.0
        for start in range(0, len(train_x), batch_size):
            batch = order[start:start + batch_size]
            logits = head(train_x[batch])                     # 1. forward pass
            loss = loss_fn(logits, train_y[batch])            # 2. how wrong is it?
            optimizer.zero_grad()                             # 3. clear old gradients
            loss.backward()                                   # 4. backpropagation: compute new gradients
            optimizer.step()                                  # 5. nudge the weights
            total += loss.item() * len(batch)

        val_loss, val_acc = evaluate(head, val_x, val_y)
        history.append(Epoch(epoch, total / len(train_x), val_loss, val_acc))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    return best_state, [asdict(e) for e in history], best_epoch


def predict_probs(head: nn.Module, x) -> np.ndarray:
    """Breed probabilities [N, 120] for cached features [N, 1408]."""
    head.eval()
    with torch.inference_mode():
        return torch.softmax(head(torch.as_tensor(x)), dim=1).numpy()


# ---------------------------------------------------------------- 5. saving and loading

def save_head(path: Path, head: BreedHead, breeds: list[str], info: dict) -> None:
    """Saves the head's weights plus what's needed to use it (the breed order) and how it was made."""
    path.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "breeds": breeds, "info": info}, path)


def load_head(path: Path) -> tuple[BreedHead, list[str], dict]:
    saved = torch.load(path, weights_only=True)   # weights_only: never run code from a file
    w = saved["state_dict"]["linear.weight"]
    head = BreedHead(in_features=w.shape[1], n_breeds=w.shape[0])
    head.load_state_dict(saved["state_dict"])
    return head.eval(), saved["breeds"], saved["info"]


# ================================================================ stage 6: unfreeze the last layers
#
# Stage 5 showed that a new head on FROZEN features can't do better: both heads read the
# same 1,408 numbers. In stage 6 the last part of the backbone is trained too, so the
# features themselves can adapt to telling dog breeds apart.
#
#   photo -> features[0..6] (frozen) -> features[7..8] (TRAINED) -> avgpool -> our head (TRAINED)
#
# Because the features now change during training, we can't cache them. Every epoch runs
# real photos through the model, so we can also use data augmentation.

def train_transform(weights):
    """Data augmentation: every epoch, each photo is randomly cropped, flipped and recoloured a little.

    The model never sees exactly the same picture twice, so it can't just memorise the
    training photos. Size and normalisation match the pre-trained model's own transform.
    """
    from torchvision import transforms as T
    eval_t = weights.transforms()
    size = eval_t.crop_size[0]
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.35, 1.0), interpolation=eval_t.interpolation),  # a random part, 35-100% of the photo
        T.RandomHorizontalFlip(),                                                           # a dog facing left is still the same breed
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),                        # different light and cameras
        T.ToTensor(),
        T.Normalize(eval_t.mean, eval_t.std),
    ])


class PartlyFrozen(nn.Module):
    """An EfficientNet with its first blocks frozen, its last blocks trainable, and our head on top."""

    def __init__(self, model: nn.Module, first_trained_block: int, head: nn.Module):
        super().__init__()
        self.first_trained_block = first_trained_block
        self.frozen = model.features[:first_trained_block]
        self.tuned = model.features[first_trained_block:]
        self.avgpool = model.avgpool
        self.head = head
        for p in self.frozen.parameters():
            p.requires_grad = False      # no gradients: these weights never change

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():            # the frozen part needs no backpropagation: saves time and memory
            x = self.frozen(x)
        return self.head(torch.flatten(self.avgpool(self.tuned(x)), 1))

    def train(self, mode: bool = True):
        """Only the head's dropout switches to training behaviour. The backbone stays in eval mode,
        so BatchNorm keeps ImageNet's statistics (our batches of 32 are too small to estimate new ones)."""
        super().train(False)
        self.head.train(mode)
        return self

    def trained_state(self) -> dict:
        """Only what changed: the tuned blocks and the head (the frozen part is plain ImageNet)."""
        return {"tuned": {k: v.detach().cpu().clone() for k, v in self.tuned.state_dict().items()},
                "head": {k: v.detach().cpu().clone() for k, v in self.head.state_dict().items()}}

    def load_trained_state(self, state: dict) -> None:
        self.tuned.load_state_dict(state["tuned"])
        self.head.load_state_dict(state["head"])


def predict_photos(model: nn.Module, dataset, device: str, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Breed probabilities [N, 120] and labels [N] for a dataset of photos."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    model.to(device).eval()
    probs, labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
            labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def train_partly_frozen(model: PartlyFrozen, train_ds, val_ds, device: str, epochs: int, lr: float,
                        batch_size: int = 32, weight_decay: float = 1e-4, seed: int = 0, on_epoch=None):
    """The same training loop as stage 5, now on photos instead of cached features.

    Returns (best_state, history, best_epoch); best = lowest validation loss (early stopping).
    on_epoch(epoch, state, history) is called after every epoch, e.g. to save a checkpoint.
    """
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    def validate():
        probs, labels = predict_photos(model, val_ds, device, batch_size)
        p = torch.as_tensor(probs).clamp_min(1e-12)
        loss = nn.functional.nll_loss(p.log(), torch.as_tensor(labels)).item()
        return loss, float((probs.argmax(1) == labels).mean())

    val_loss, val_acc = validate()
    history = [asdict(Epoch(0, float("nan"), val_loss, val_acc))]
    print(f"  epoch  0: val loss {val_loss:.3f}, val accuracy {val_acc:.1%}", flush=True)
    best_loss, best_epoch, best_state = val_loss, 0, model.trained_state()

    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)            # 1-2. forward pass and loss
            optimizer.zero_grad()                  # 3. clear old gradients
            loss.backward()                        # 4. backpropagation (through the tuned blocks and head only)
            optimizer.step()                       # 5. nudge the trainable weights
            total += loss.item() * len(y)
            seen += len(y)
            if i % 50 == 0:
                print(f"    epoch {epoch}: {seen:>6} / {len(train_ds)} photos, train loss so far {total / seen:.3f}",
                      flush=True)
        val_loss, val_acc = validate()
        history.append(asdict(Epoch(epoch, total / seen, val_loss, val_acc)))
        print(f"  epoch {epoch:2d}: train loss {total / seen:.3f}, val loss {val_loss:.3f}, "
              f"val accuracy {val_acc:.1%}", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, model.trained_state()
        if on_epoch:
            on_epoch(epoch, best_state, history, best_epoch)
    return best_state, history, best_epoch
