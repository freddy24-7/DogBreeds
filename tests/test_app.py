"""Checks that every stage tab can be built and that the stages are set up correctly."""
from dogbreeds.stages import STAGES


def test_stage_ids_are_unique_and_numbered():
    assert [s.ID for s in STAGES] == [f"stage{i}" for i in range(1, len(STAGES) + 1)]


def test_app_builds_all_tabs():
    import app  # building the Blocks runs every stage's build()
    assert app.demo is not None


def test_every_stage_is_in_the_stage_picker():
    import app
    assert {stage_id for _, stage_id in app.STAGE_CHOICES} == {s.ID for s in STAGES}
    assert app.STAGE_CHOICES[0][1] == STAGES[-1].ID          # the newest stage comes first


def test_choosing_a_stage_shows_only_that_page():
    import app
    updates = app.show_stage("stage3")
    assert [u["visible"] for u in updates] == [s.ID == "stage3" for s in STAGES]
