"""Synthetic EEG with an injected seizure-like burst, for screening adaptive-filter settings.

The background is 1/f ("pink") noise plus a 10 Hz alpha-like resonance, scaled to a requested RMS
in microvolts. The burst is either a 3 Hz spike-and-wave train that slows slightly, or a rhythmic
chirp (6 -> 12 Hz with a second harmonic); it is scaled so that the RMS of (background + burst) over
the burst interval is exactly ``ratio`` times the background RMS. Everything is seeded.

This is a controlled test bench, not a model of real seizures: it answers "would this filter erase a
burst with this amplitude, shape and duration?", nothing more.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

KINDS = ("spikewave", "rhythmic")


def _unit_rms(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True))


def background(n_channels: int, n_samples: int, fs: float, rms: float, rng: np.random.Generator) -> np.ndarray:
    """1/f noise plus a 10 Hz resonance, each channel scaled to ``rms``."""
    white = rng.normal(size=(n_channels, n_samples))
    spectrum = np.fft.rfft(white, axis=1)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)
    freqs[0] = freqs[1]
    pink = _unit_rms(np.fft.irfft(spectrum / np.sqrt(freqs), n=n_samples, axis=1))
    radius, omega = 0.98, 2 * np.pi * 10.0 / fs
    alpha = _unit_rms(lfilter([1.0], [1.0, -2 * radius * np.cos(omega), radius**2], rng.normal(size=(n_channels, n_samples)), axis=1))
    return rms * _unit_rms(0.8 * pink + 0.5 * alpha)


def _envelope(n: int, fs: float) -> np.ndarray:
    ramp = max(2, min(int(fs), n // 5))
    env = np.ones(n)
    shape = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp)))
    env[:ramp] = shape
    env[-ramp:] = shape[::-1]
    return env


def burst_waveform(kind: str, seconds: float, fs: float) -> np.ndarray:
    """Unit-RMS burst of the given kind and duration (1-D)."""
    if kind not in KINDS:
        raise ValueError(f"unknown burst kind '{kind}'; choose from {KINDS}")
    n = int(round(seconds * fs))
    if kind == "spikewave":
        freq = np.linspace(3.2, 2.6, n)  # slows down a little, as many seizures do
        phase = np.cumsum(freq) / fs % 1.0
        wave = np.exp(-0.5 * ((phase - 0.15) / 0.025) ** 2) - 0.5 * np.exp(-0.5 * ((phase - 0.55) / 0.12) ** 2)
    else:
        freq = np.linspace(6.0, 12.0, n)
        phase = 2 * np.pi * np.cumsum(freq) / fs
        wave = np.sin(phase) + 0.3 * np.sin(2 * phase + 0.7)
    wave = (wave - wave.mean()) * _envelope(n, fs)
    return wave / np.sqrt(np.mean(wave**2))


def make_recording(
    burst_s: float,
    kind: str = "spikewave",
    fs: float = 256.0,
    pre_s: float = 300.0,
    post_s: float = 60.0,
    ratio: float = 4.0,
    n_channels: int = 4,
    rms: float = 20.0,
    seed: int = 0,
) -> dict:
    """Background with one burst. Returns ``{"x", "onset", "end", "fs"}`` (sample indices)."""
    if ratio <= 1.0:
        raise ValueError("ratio must be > 1")
    rng = np.random.default_rng(seed)
    onset = int(round(pre_s * fs))
    length = int(round(burst_s * fs))
    total = onset + length + int(round(post_s * fs))
    x = background(n_channels, total, fs, rms, rng)
    wave = burst_waveform(kind, burst_s, fs)
    for c in range(n_channels):
        seg = x[c, onset : onset + length]
        bg_power = np.mean(seg**2)
        cross = np.mean(seg * wave)
        # solve mean((seg + a*wave)^2) = ratio^2 * rms^2 for a > 0
        b = cross
        disc = b * b + (ratio**2 * rms**2 - bg_power)
        scale = -b + np.sqrt(max(disc, 0.0))
        x[c, onset : onset + length] = seg + scale * wave
    return {"x": x, "onset": onset, "end": onset + length, "fs": fs}
