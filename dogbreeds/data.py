"""
Datasets: Stanford Dogs (stage 4 on), Oxford-IIIT Pets dogs, and your own photos.

All three give the same thing: (image tensor, breed number), with breed numbers
0..119 for the 120 Stanford Dogs breeds. So the same evaluation code (and in
stage 5 the same training code) works on all of them.

Where the data lives (not in git, see README_stage4.md):
    data/stanford-dogs/Images/n02085620-Chihuahua/n02085620_2650.jpg ...
    data/stanford-dogs/train_list.mat, test_list.mat     (the official split)
    data/oxford-iiit-pet/                                (stage 3)
    data/my_photos/<breed name>/<any name>.jpg           (optional, your own photos)
"""
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STANFORD = DATA / "stanford-dogs"
PETS = DATA / "oxford-iiit-pet"
MY_PHOTOS = DATA / "my_photos"


def _norm(name: str) -> str:
    """'black-and-tan_coonhound' and 'black-and-tan coonhound' -> 'blackandtancoonhound'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass(frozen=True)
class Breed:
    index: int            # 0..119, our own breed number
    name: str             # e.g. "Siberian husky" (ImageNet spelling)
    folder: str           # e.g. "n02110185-Siberian_husky"
    imagenet_index: int   # the same breed's class number in ImageNet (0..999)


def stanford_breeds(imagenet_categories: list[str]) -> list[Breed]:
    """The 120 Stanford Dogs breeds, matched to their ImageNet class by name.

    117 are ImageNet's dog breeds (151-268; only 'dalmatian' is missing) and 3 are
    wild dogs just after them: dingo (273), dhole (274), African hunting dog (275).
    """
    from scipy.io import loadmat  # only needed here

    folders = sorted({str(f[0][0]).split("/")[0] for f in loadmat(STANFORD / "file_list.mat")["file_list"]})
    by_name: dict[str, list[int]] = {}
    for i, c in enumerate(imagenet_categories):
        by_name.setdefault(_norm(c), []).append(i)

    breeds = []
    for index, folder in enumerate(folders):
        matches = by_name[_norm(folder.split("-", 1)[1])]
        # "Cardigan" matches both the corgi (264) and the sweater (474): keep the dog.
        matches = [m for m in matches if 151 <= m <= 275]
        assert len(matches) == 1, f"{folder}: {matches}"
        breeds.append(Breed(index, imagenet_categories[matches[0]], folder, matches[0]))
    return breeds


class PhotoDataset(Dataset):
    """A list of (image path, breed number) pairs. Transform is applied when an image is read."""

    def __init__(self, items: list[tuple[Path, int]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        image = Image.open(path).convert("RGB")
        return (self.transform(image) if self.transform else image), label


def stanford_dogs(split: str, breeds: list[Breed], transform=None) -> PhotoDataset:
    """The official Stanford Dogs split: 'train' (12,000 photos) or 'test' (8,580)."""
    from scipy.io import loadmat

    folder_to_index = {b.folder: b.index for b in breeds}
    files = loadmat(STANFORD / f"{split}_list.mat")["file_list"]
    items = []
    for f in files:
        rel = str(f[0][0])
        items.append((STANFORD / "Images" / rel, folder_to_index[rel.split("/")[0]]))
    return PhotoDataset(items, transform)


# Oxford-IIIT Pets dog breeds -> ImageNet name (as in stage 3). Four Oxford breeds have
# no Stanford/ImageNet equivalent and are left out: American bulldog, American pit bull
# terrier, Havanese and Shiba Inu.
OXFORD_TO_IMAGENET = {
    "basset_hound": "basset", "beagle": "beagle", "boxer": "boxer", "chihuahua": "Chihuahua",
    "english_cocker_spaniel": "cocker spaniel", "english_setter": "English setter",
    "german_shorthaired": "German short-haired pointer", "great_pyrenees": "Great Pyrenees",
    "japanese_chin": "Japanese spaniel", "keeshond": "keeshond", "leonberger": "Leonberg",
    "miniature_pinscher": "miniature pinscher", "newfoundland": "Newfoundland",
    "pomeranian": "Pomeranian", "pug": "pug", "saint_bernard": "Saint Bernard", "samoyed": "Samoyed",
    "scottish_terrier": "Scotch terrier", "staffordshire_bull_terrier": "Staffordshire bullterrier",
    "wheaten_terrier": "soft-coated wheaten terrier", "yorkshire_terrier": "Yorkshire terrier",
}


def oxford_dogs(split: str, breeds: list[Breed], transform=None) -> PhotoDataset:
    """Oxford-IIIT Pets dog photos of the 21 breeds that Stanford Dogs also has."""
    name_to_index = {b.name: b.index for b in breeds}
    items = []
    for line in (PETS / "annotations" / f"{split}.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, species, _ = line.split()
        breed = OXFORD_TO_IMAGENET.get(name.rsplit("_", 1)[0])
        if species == "2" and breed:
            items.append((PETS / "images" / f"{name}.jpg", name_to_index[breed]))
    return PhotoDataset(items, transform)


def my_photos(breeds: list[Breed], transform=None) -> PhotoDataset | None:
    """Your own photos in data/my_photos/<breed>/, if any. Folder names are matched loosely,
    so 'golden retriever', 'golden_retriever' and 'Golden-Retriever' all work."""
    if not MY_PHOTOS.exists():
        return None
    by_name = {_norm(b.name): b.index for b in breeds} | {_norm(b.folder.split("-", 1)[1]): b.index for b in breeds}
    items, unknown = [], []
    for folder in sorted(p for p in MY_PHOTOS.iterdir() if p.is_dir()):
        index = by_name.get(_norm(folder.name))
        if index is None:
            unknown.append(folder.name)
            continue
        items += [(p, index) for p in sorted(folder.iterdir())
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]  # iPhone HEIC: export as JPEG first
    if unknown:
        print(f"my_photos: skipped folders that aren't one of the 120 breeds: {unknown}")
    return PhotoDataset(items, transform) if items else None
