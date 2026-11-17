"""Checks that every stage tab can be built and that the stages are set up correctly."""
from dogbreeds.stages import STAGES


def test_stage_ids_are_unique_and_numbered():
    assert [s.ID for s in STAGES] == [f"stage{i}" for i in range(1, len(STAGES) + 1)]


def test_app_builds_all_tabs():
    import app  # building the Blocks runs every stage's build()
    assert app.demo is not None
