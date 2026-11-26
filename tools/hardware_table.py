"""Compute-cost and latency table for the deployment-oriented study (roadmap 7.1 and 7.2).

Everything except the timing column is analytic (counted from the code in
``src/eegpipe/adaptive``), so it needs no data. Timing is measured on the machine that runs
this script (Numba kernels, compile call excluded), so quote it as "measured on <machine>".

Usage (from the repo root):
    python tools/hardware_table.py
    python tools/hardware_table.py --no-timing          # analytic columns only (fast)
    python tools/hardware_table.py --seconds 120

Outputs: results/tables/hardware_table.csv and results/tables/latency_budget.csv, plus a
printed Markdown version of both.

Counting conventions
--------------------
* "as implemented" counts the operations the current Numba kernels execute per input sample per
  channel. A multiply-accumulate counts as one multiply and one add.
* "minimal" is what a straightforward hardware implementation would need after obvious
  simplifications (recursive power update for NLMS, reciprocals computed once for RLS, the gain
  exponential computed once per block for the Kalman stage). These are estimates of an
  optimised implementation, not measurements.
* State bytes assume float64 (as in the code); the float32 column halves it.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from eegpipe.adaptive.kalman_gain import KalmanGainTracker
from eegpipe.adaptive.nlms import NLMSFilter
from eegpipe.adaptive.rls import RLSFilter
from eegpipe.config import load_config

FS = 256
CHANNELS = 18


@dataclass(frozen=True)
class Ops:
    """Per-sample, per-channel operation counts."""

    mul: float
    add: float
    div: float = 0.0
    transcendental: float = 0.0  # exp / log

    @property
    def total(self) -> float:
        return self.mul + self.add + self.div + self.transcendental


def nlms_ops(length: int, minimal: bool = False) -> Ops:
    """NLMS: prediction, power, update. Minimal uses a recursive power (add new, drop old)."""
    if minimal:
        # y: L; power recursion: 2 mul + 2 add; error 1; gain: 1 mul + 1 add + 1 div; w: L
        return Ops(mul=2 * length + 3, add=2 * length + 4, div=1)
    # y: L, power: L, gain: 1 add + 1 mul + 1 div, error: 1 sub, update: L
    return Ops(mul=3 * length + 1, add=3 * length + 2, div=1)


def rls_ops(length: int, minimal: bool = False) -> Ops:
    """RLS with the symmetric (upper-triangle) P update used in ``rls.py``."""
    tri = length * (length + 1) // 2
    # y (L), power (L), P u (L^2), denominator (L), weight update (L mul + L div + L add),
    # P update: per upper entry 1 mul, 2 div, 1 sub
    mul = length + length + length**2 + length + length + tri
    add = length + length + length**2 + length + length + 1 + tri
    div = length + 2 * tri
    if minimal:
        # reciprocal of denom and of lambda computed once; every division becomes a multiply
        return Ops(mul=mul + div - 2, add=add, div=2)
    return Ops(mul=mul, add=add, div=div)


def kalman_ops(block: int, minimal: bool = False) -> Ops:
    """Kalman gain tracker: per sample x*g and the sum of squares; per block a log and an update."""
    per_block_mul, per_block_add, per_block_div = 3, 6, 2
    exp_per_sample = 1.0 / block if minimal else 1.0  # the code evaluates exp(-s) per sample
    return Ops(
        mul=2 + per_block_mul / block,
        add=2 + per_block_add / block,
        div=per_block_div / block,
        transcendental=exp_per_sample + 1.0 / block,
    )


def fir_ops(numtaps: int, symmetric: bool = False) -> Ops:
    """Direct-form FIR. A linear-phase FIR is symmetric, so only half the multiplies are needed."""
    mul = (numtaps + 1) // 2 if symmetric else numtaps
    return Ops(mul=mul, add=numtaps - 1)


def biquad_ops() -> Ops:
    """One direct-form-I biquad (the 60 Hz notch): 5 multiplies, 4 adds."""
    return Ops(mul=5, add=4)


def state_doubles(method: str, length: int = 0) -> int:
    if method == "nlms":
        return 2 * length
    if method == "rls":
        return 2 * length + length * length
    if method == "kalman":
        return 5
    if method == "fir":
        return length - 1  # delay line of numtaps - 1 samples
    if method == "biquad":
        return 4
    raise ValueError(method)


def _time_per_sample(make, data: np.ndarray) -> float:
    make().process(data[:, : 2 * FS])  # compile and warm up on a throw-away instance
    filt = make()
    start = time.perf_counter()
    filt.process(data)
    return (time.perf_counter() - start) / data.size


def build_table(seconds: float, timing: bool, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    numtaps, _ = fir_numtaps_and_delay(cfg)
    rng = np.random.default_rng(0)
    white = rng.normal(0.0, 10.0, (CHANNELS, int(seconds * FS)))
    data = np.cumsum(white, axis=1) * 0.05 + white
    block = FS  # 1 s blocks, as in the frozen Kalman setting

    entries = [
        ("Kalman gain tracker (frozen primary)", "kalman", 0, kalman_ops(block), kalman_ops(block, True),
         lambda: KalmanGainTracker(CHANNELS, fs=FS, block_s=1.0, q_over_r=3e-5)),
        ("NLMS L=8", "nlms", 8, nlms_ops(8), nlms_ops(8, True), lambda: NLMSFilter(CHANNELS, length=8, mu=0.005)),
        ("NLMS L=16", "nlms", 16, nlms_ops(16), nlms_ops(16, True), lambda: NLMSFilter(CHANNELS, length=16, mu=0.005)),
        ("RLS L=8", "rls", 8, rls_ops(8), rls_ops(8, True), lambda: RLSFilter(CHANNELS, length=8, forgetting=0.9999)),
        ("RLS L=16", "rls", 16, rls_ops(16), rls_ops(16, True), lambda: RLSFilter(CHANNELS, length=16, forgetting=0.9999)),
    ]
    # Fixed preprocessing stages, for comparison with the adaptive stage (no timing: the offline
    # pipeline applies them with filtfilt / FFT convolution, which is not a per-sample kernel).
    entries += [
        (f"Preprocessing: Kaiser FIR band-pass, {numtaps} taps", "fir", numtaps, fir_ops(numtaps),
         fir_ops(numtaps, symmetric=True), None),
        ("Preprocessing: 60 Hz notch (biquad)", "biquad", 0, biquad_ops(), biquad_ops(), None),
    ]
    rows = []
    for name, method, length, ops, ops_min, make in entries:
        doubles = state_doubles(method, length)
        row = {
            "stage": name,
            "mul_per_sample": round(ops.mul, 2),
            "add_per_sample": round(ops.add, 2),
            "div_per_sample": round(ops.div, 3),
            "exp_log_per_sample": round(ops.transcendental, 3),
            "ops_per_sample_as_implemented": round(ops.total, 1),
            "ops_per_sample_minimal": round(ops_min.total, 1),
            "div_per_sample_minimal": round(ops_min.div, 3),
            "mops_per_s_18ch_as_implemented": round(ops.total * FS * CHANNELS / 1e6, 3),
            "mops_per_s_18ch_minimal": round(ops_min.total * FS * CHANNELS / 1e6, 3),
            "state_bytes_per_channel_f64": doubles * 8,
            "state_bytes_18ch_f32": doubles * 4 * CHANNELS,
        }
        if timing and make is not None:
            per_sample = _time_per_sample(make, data)
            row["measured_us_per_sample"] = round(per_sample * 1e6, 4)
            row["x_real_time_1core"] = round(1.0 / (per_sample * FS * CHANNELS))
        rows.append(row)
    return pd.DataFrame(rows)


def fir_numtaps_and_delay(cfg: dict) -> tuple[int, float]:
    prep = cfg["preprocessing"]
    ripple, transition = prep["fir"]["ripple_db"], prep["fir"]["transition_hz"]
    numtaps, _ = signal.kaiserord(ripple, transition / (FS / 2.0))
    numtaps += int(numtaps % 2 == 0)
    return numtaps, (numtaps - 1) / 2.0 / FS


def latency_budget(cfg: dict, min_consecutive: int = 3) -> pd.DataFrame:
    seg = cfg["segmentation"]
    window_s = float(seg["window_s"])
    step_s = window_s * (1.0 - float(seg["overlap"]))
    numtaps, fir_delay = fir_numtaps_and_delay(cfg)
    extra = (min_consecutive - 1) * step_s
    rows = [
        ("Window length (decision needs the whole window)", window_s, f"{window_s:.0f} s windows"),
        (f"Extra windows before alarm (min_consecutive={min_consecutive})", extra,
         f"hop {step_s:.0f} s; the frozen setting needs {min_consecutive} consecutive positive windows"),
        ("Kalman estimate lag (not an output delay)", 0.0,
         "the gain estimate uses blocks before the current one; output samples are not delayed"),
        ("Structural minimum, as evaluated (offline, zero-phase preprocessing)", window_s + extra,
         "plus feature and classifier time; the measured median event latency is higher"),
        ("Causal FIR band-pass group delay (needed only if deployed)", fir_delay,
         f"{numtaps} taps, linear phase; the offline code removes this delay with a centred filter"),
        ("Structural minimum, if deployed with a causal FIR", window_s + extra + fir_delay,
         "the notch would add a few samples more; not counted"),
    ]
    return pd.DataFrame(rows, columns=["component", "seconds", "note"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seconds", type=float, default=60.0, help="length of the timing signal")
    parser.add_argument("--no-timing", action="store_true", help="skip the measured column")
    parser.add_argument("--min-consecutive", type=int, default=3)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["tables"])
    out_dir.mkdir(parents=True, exist_ok=True)

    table = build_table(args.seconds, timing=not args.no_timing, cfg=cfg)
    budget = latency_budget(cfg, args.min_consecutive)
    table.to_csv(out_dir / "hardware_table.csv", index=False)
    budget.to_csv(out_dir / "latency_budget.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(table.to_string(index=False))
    print()
    print(budget.to_string(index=False))
    print(f"\nWrote {out_dir / 'hardware_table.csv'} and {out_dir / 'latency_budget.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
