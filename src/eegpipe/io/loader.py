"""EDF reading: header inspection and low-memory, name-based channel loading.

Real CHB-MIT files contain placeholder channels named ``-``, sometimes a duplicated
``T8-P8`` label (MNE renames duplicates to ``T8-P8-0`` / ``T8-P8-1``) and channels in varying
order, so channels are always selected **by name** and returned in the configured order.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import mne
import numpy as np

from eegpipe.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ChannelMissingError(Exception):
    """A required channel is not present in an EDF file."""


def _read_raw(edf_path: str | Path):
    """Lazy MNE read. Silences only the two expected header warnings (handled deliberately):
    duplicate channel labels (renamed to ``-0``/``-1`` by MNE) and missing measurement date."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Channel names are not unique")
        warnings.filterwarnings("ignore", message="Invalid measurement date")
        return mne.io.read_raw_edf(edf_path, preload=False, verbose=False)


def read_edf_header(edf_path: str | Path) -> dict:
    """Read duration and channel names without loading the signal.

    ``n_samples`` comes from ``raw.n_times`` (``raw.times[-1]`` would be one sample short).
    """
    raw = _read_raw(edf_path)
    sfreq = float(raw.info["sfreq"])
    return {
        "n_samples": int(raw.n_times),
        "duration_s": raw.n_times / sfreq if sfreq > 0 else 0.0,
        "sfreq": sfreq,
        "ch_names": list(raw.ch_names),
    }


def resolve_channel_names(
    available: list[str], required: list[str], aliases: dict[str, str]
) -> dict[str, str]:
    """Map every required canonical channel to the name it has inside the file.

    Order of preference: exact name, configured alias (raw name -> canonical), then MNE's
    duplicate suffix ``<name>-0``. Raises :class:`ChannelMissingError` naming the first
    channel that cannot be resolved.
    """
    inverse: dict[str, list[str]] = {}
    for raw_name, canonical in aliases.items():
        inverse.setdefault(canonical, []).append(raw_name)
    present = set(available)
    resolved: dict[str, str] = {}
    for channel in required:
        if channel in present:
            resolved[channel] = channel
            continue
        alias_hit = next((n for n in inverse.get(channel, []) if n in present), None)
        if alias_hit is not None:
            resolved[channel] = alias_hit
            continue
        if f"{channel}-0" in present:
            logger.warning("Using duplicate-suffixed '%s-0' for channel %s", channel, channel)
            resolved[channel] = f"{channel}-0"
            continue
        raise ChannelMissingError(f"Missing channel: {channel}")
    return resolved


def check_edf(
    edf_path: str | Path,
    channels: list[str],
    aliases: dict[str, str],
    fs_expected: int = 256,
) -> tuple[bool, str, dict]:
    """Header-only usability check. Returns ``(ok, reason, header)``.

    ``reason`` is empty when ``ok``; otherwise ``unreadable_edf:<error type>``,
    ``bad_fs:<value>`` or ``missing_channel:<name>``.
    """
    try:
        header = read_edf_header(edf_path)
    except Exception as exc:  # corrupt or truncated file
        return False, f"unreadable_edf:{type(exc).__name__}", {}
    if abs(header["sfreq"] - fs_expected) > 1e-6:
        return False, f"bad_fs:{header['sfreq']:g}", header
    try:
        resolve_channel_names(header["ch_names"], channels, aliases)
    except ChannelMissingError as exc:
        return False, f"missing_channel:{str(exc).split(': ', 1)[1]}", header
    return True, "", header


def load_edf_channels(
    edf_path: str | Path,
    channels: list[str],
    aliases: dict[str, str],
    fs_expected: int = 256,
    chunk_s: float = 600.0,
) -> tuple[np.ndarray, float]:
    """Load ``channels`` (in that order) as a float32 array in microvolts.

    The file is read lazily in ``chunk_s`` blocks straight into a preallocated float32 array,
    so peak memory is about the size of the output instead of several float64 copies of the
    whole recording (this matters for the 4-hour CHB-MIT files).

    Returns ``(data, fs)`` with ``data`` of shape ``(len(channels), n_samples)``.
    """
    raw = _read_raw(edf_path)
    fs = float(raw.info["sfreq"])
    if abs(fs - fs_expected) > 1e-6:
        raise ValueError(f"Expected fs={fs_expected}, got {fs} in {Path(str(edf_path)).name}")
    resolved = resolve_channel_names(list(raw.ch_names), channels, aliases)
    picks = [raw.ch_names.index(resolved[c]) for c in channels]
    unique = sorted(set(picks))
    column_of = {p: i for i, p in enumerate(unique)}
    selector = [column_of[p] for p in picks]

    n_times = int(raw.n_times)
    out = np.empty((len(channels), n_times), dtype=np.float32)
    step = max(1, int(chunk_s * fs))
    for start in range(0, n_times, step):
        stop = min(start + step, n_times)
        block = raw.get_data(picks=unique, start=start, stop=stop)  # volts, float64
        out[:, start:stop] = block[selector] * 1e6  # -> microvolts
    return out, fs
