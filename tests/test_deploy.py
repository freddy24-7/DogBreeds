"""Stage 9: the package has everything the app needs, with the same versions as here, and was measured."""
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import stage9_deploy as deploy  # noqa: E402


def test_every_photo_the_pages_show_exists():
    photos = deploy.referenced_photos()
    assert photos and all((ROOT / p).exists() for p in photos)


def test_requirements_pin_the_versions_we_test_with():
    req = deploy.requirements()
    for package in ["torch", "torchvision", "numpy", "pillow", "scipy"]:
        assert f"{package}=={metadata.version(package)}\n" in req
    assert "gradio" not in req.replace("# gradio", "")        # the Space installs gradio itself


def test_space_readme_sets_the_gradio_version_and_app_file():
    head = deploy.SPACE_README.split("---")[1]
    assert f"sdk_version: {metadata.version('gradio')}" in head and "app_file: app.py" in head


def test_the_app_uses_the_port_the_space_gives_it():
    assert 'os.environ.get("GRADIO_SERVER_PORT", 7870)' in (ROOT / "app.py").read_text()


@pytest.mark.skipif(not (ROOT / "reports" / "stage9_report.json").exists(), reason="run scripts/stage9_deploy.py")
def test_stage9_report_has_the_server_measurements():
    r = json.loads((ROOT / "reports" / "stage9_report.json").read_text())
    m = r["measured"]
    assert 100 < m["peak_memory_megabytes"] < 8000 and m["start_seconds"] > 0
    assert all(0 < s < 30 for s in m["seconds_per_photo"].values())
    assert abs(sum(r["parts_megabytes"].values()) - r["bundle_megabytes"]) < 1   # the parts add up to the total
