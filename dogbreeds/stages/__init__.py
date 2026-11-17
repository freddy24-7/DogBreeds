"""
One module per stage. Each builds its own tab in the app, so you can open any
stage and see how the app looked and behaved at that point.

Every stage module has:
    ID     short id, used to select the tab
    TITLE  the tab's label
    build()  creates the tab's contents (called inside a gr.Tab)

To add a stage: write stageN.py with these three and add it to STAGES.
"""
from dogbreeds.stages import stage1, stage2

STAGES = [stage1, stage2]
