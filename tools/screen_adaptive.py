"""Synthetic seizure-preservation screen of every adaptive setting in the pilot grid (chunk 2.6).

For each setting of ``adaptive.pilot_grid`` it injects spike-wave and rhythmic bursts of several
durations into synthetic EEG-like background, filters the recording, and reports how much of the
burst's feature separability survives (pass rule, default >= 0.85 for 15, 30 and 60 s bursts of both
kinds). No patient data is touched. See src/eegpipe/adaptive/preservation.py.

Usage (from the repo root):
    python tools/screen_adaptive.py                       # whole pilot grid, 3 seeds
    python tools/screen_adaptive.py --methods kalman      # one method
    python tools/screen_adaptive.py --seeds 0 --lengths 15 30 --threshold 0.85

Writes results/tables/adaptive_screen_summary.csv and adaptive_screen_detail.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from eegpipe.adaptive import METHODS, expand_pilot_grid, warm_start_from_config
from eegpipe.adaptive.preservation import DEFAULT_LENGTHS, RULE_LENGTHS, THRESHOLD, screen
from eegpipe.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--lengths", nargs="+", type=float, default=list(DEFAULT_LENGTHS),
                        help="burst durations in seconds")
    parser.add_argument("--rule-lengths", nargs="+", type=float, default=list(RULE_LENGTHS))
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    settings = [(m, p) for m in args.methods for p in expand_pilot_grid(cfg, m)]
    if not settings:
        raise SystemExit("The config has no adaptive.pilot_grid for the chosen methods.")
    print(f"Screening {len(settings)} settings x {len(args.lengths)} burst lengths x 2 kinds x "
          f"{len(args.seeds)} seed(s)...")

    started = time.time()
    detail, summary = screen(
        settings,
        burst_lengths=tuple(args.lengths),
        seeds=tuple(args.seeds),
        threshold=args.threshold,
        rule_lengths=tuple(args.rule_lengths),
        warm_start_s=warm_start_from_config(cfg),
    )
    out_dir = Path(cfg["paths"]["tables"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "adaptive_screen_summary.csv", index=False)
    detail.to_csv(out_dir / "adaptive_screen_detail.csv", index=False)

    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    passed = int(summary["passes"].sum())
    print(f"\n{passed} of {len(summary)} settings pass (threshold {args.threshold}); "
          f"{time.time() - started:.0f} s. Wrote tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
