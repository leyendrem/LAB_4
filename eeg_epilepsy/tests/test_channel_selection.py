import json
from eeg_epilepsy.data.channel_selection import select_analysis_channels

def test_exclusions_are_applied_without_mutating_original_names(tmp_path):
    config = {"excluded_channels":[{"name":"T8-P8-1","reason":"duplicate"},{"name":"P7-T7","reason":"redundant"}]}
    p = tmp_path / "channel_selection.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    original = ["FP1-F7", "T8-P8-0", "T8-P8-1", "T7-P7", "P7-T7"]
    selected = select_analysis_channels(original, p)
    assert selected == ["FP1-F7", "T8-P8-0", "T7-P7"]
    assert original == ["FP1-F7", "T8-P8-0", "T8-P8-1", "T7-P7", "P7-T7"]
