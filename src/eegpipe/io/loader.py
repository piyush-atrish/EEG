import numpy as np
import mne
from pathlib import Path
from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ChannelMissingError(Exception):
    """Raised when an EDF file is missing one of the 18 required channels."""
    pass

def read_edf_header(edf_path: Path) -> dict:
    """Read duration and channel names without preloading data."""
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    duration_s = raw.times[-1] if len(raw.times) > 0 else 0.0
    return {
        "duration_s": duration_s,
        "ch_names": raw.ch_names
    }

def load_edf_channels(edf_path: Path, channels: list[str], aliases: dict, fs_expected: int = 256) -> tuple[np.ndarray, float]:
    """Load EDF, resolve aliases, pick channels in order, and return float32 uV array."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    if raw.info["sfreq"] != fs_expected:
        raise ValueError(f"Expected fs={fs_expected}, got {raw.info['sfreq']} in {edf_path.name}")

    # Resolve aliases
    rename_dict = {ch: aliases[ch] for ch in raw.ch_names if ch in aliases}
    if rename_dict:
        raw.rename_channels(rename_dict)

    # Validate channel presence
    missing = [ch for ch in channels if ch not in raw.ch_names]
    if missing:
        raise ChannelMissingError(f"Missing channel: {missing[0]}")

    # Pick and strictly reorder channels to match config order
    raw.pick(channels)
    raw.reorder_channels(channels)
    
    # MNE stores EEG data in Volts. Convert to microvolts (uV)
    data_uV = raw.get_data() * 1e6
    
    return data_uV.astype(np.float32), raw.info["sfreq"]