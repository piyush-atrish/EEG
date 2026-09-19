import pytest
from eegpipe.config import load_config, channel_slug

def test_load_config_validates_channels():
    # Pass an override with only 1 channel to trigger the validation error
    overrides = {"dataset": {"channels": ["FP1-F7"]}}
    with pytest.raises(ValueError, match="exactly 18 unique channels"):
        load_config(overrides=overrides)

def test_load_config_overrides():
    cfg = load_config(overrides={"project": {"seed": 99}})
    assert cfg["project"]["seed"] == 99

def test_channel_slug():
    assert channel_slug("FP1-F7") == "FP1_F7"
    assert channel_slug("T8-P8-0") == "T8_P8_0"