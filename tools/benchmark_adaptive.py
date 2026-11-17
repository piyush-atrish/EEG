"""Measure the speed of the adaptive filters and project the time for the whole cohort.

Usage (from the repo root):
    python tools/benchmark_adaptive.py
    python tools/benchmark_adaptive.py --seconds 120 --hours 125

The first call of each filter compiles its Numba kernel; that call is excluded from the timing.
The projection assumes one core and 18 channels at 256 Hz. Cases can run in parallel, so divide by
the number of workers for an estimate of wall time.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from eegpipe.adaptive.nlms import NLMSFilter
from eegpipe.adaptive.rls import RLSFilter

FS, CHANNELS = 256, 18


def make_signal(seconds: float, seed: int = 0) -> np.ndarray:
    """Coloured random test signal (smoothed noise), channels x samples, microvolt-like scale."""
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 10.0, (CHANNELS, int(seconds * FS)))
    return np.cumsum(white, axis=1) * 0.05 + white


def time_filter(make, data: np.ndarray) -> float:
    """Seconds per sample (per channel-sample) after a compile warm-up."""
    make().process(data[:, : 2 * FS])  # compile + warm up on a throw-away instance
    filt = make()
    start = time.perf_counter()
    filt.process(data)
    return (time.perf_counter() - start) / data.size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seconds", type=float, default=60.0, help="length of the test signal")
    parser.add_argument("--hours", type=float, default=125.0, help="hours of EEG to project to")
    args = parser.parse_args(argv)

    data = make_signal(args.seconds)
    total_samples = args.hours * 3600 * FS * CHANNELS
    candidates = {
        "nlms L=16": lambda: NLMSFilter(CHANNELS, length=16, mu=0.005),
        "nlms L=32": lambda: NLMSFilter(CHANNELS, length=32, mu=0.005),
        "rls  L=8": lambda: RLSFilter(CHANNELS, length=8, forgetting=0.9995),
        "rls  L=16": lambda: RLSFilter(CHANNELS, length=16, forgetting=0.9995),
    }
    print(f"{'filter':<10} {'us/sample':>10} {'x real time':>12} {'cohort (1 core)':>16}")
    realtime_samples = FS * CHANNELS  # samples per second of recording
    for name, make in candidates.items():
        per_sample = time_filter(make, data)
        speed = 1.0 / (per_sample * realtime_samples)
        minutes = per_sample * total_samples / 60.0
        print(f"{name:<10} {per_sample * 1e6:>10.3f} {speed:>11.0f}x {minutes:>12.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
