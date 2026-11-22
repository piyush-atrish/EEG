"""Pilot stage A: screen adaptive settings on the DEVELOPMENT patients with cheap features.

For each setting folder under data/interim/preprocessed_adaptive (made by scripts/02b_adaptive.py for the
development patients) and for the raw arrays, computes cheap window features and the diagnostics M1
(patient variance share, lower is better), M2 (seizure separability kept, relative to raw) and direction
agreement. Settings with M1 improved and M2 kept >= 0.9 of raw are flagged ``proceeds``; those go on to
stage B (full features + LOSO). No classifier is trained.

Usage (from the repo root):
    python tools/pilot_stage_a.py
    python tools/pilot_stage_a.py --settings nlms_L16_mu0.005 rls_L8_lam0.9999 --cap 2000

Only chb01-chb05 are allowed (frozen development/evaluation split); --allow-non-dev overrides this
for debugging and must never be used for selecting a setting.
Writes results/tables/pilot_stage_a.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from eegpipe.adaptive import adaptive_root, list_settings
from eegpipe.adaptive.pilot import DEV_PATIENTS, M2_FLOOR, missing_arrays, run_stage_a
from eegpipe.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--settings", nargs="+", default=None, help="default: every setting folder found")
    parser.add_argument("--patients", nargs="+", default=list(DEV_PATIENTS))
    parser.add_argument("--allow-non-dev", action="store_true")
    parser.add_argument("--cap", type=int, default=3000, help="interictal windows sampled per case")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--m2-floor", type=float, default=M2_FLOOR)
    parser.add_argument("--breakdown", action="store_true",
                        help="also show separability per feature type and write pilot_stage_a_by_feature.csv")
    args = parser.parse_args(argv)

    outside = sorted(set(args.patients) - set(DEV_PATIENTS))
    if outside and not args.allow_non_dev:
        raise SystemExit(f"{outside} are evaluation patients; the pilot may only use {list(DEV_PATIENTS)}.")

    cfg = load_config(args.config)
    explicit = bool(args.settings)
    names = args.settings or list_settings(cfg)
    if not names:
        raise SystemExit("No adaptive settings found. Run scripts/02b_adaptive.py first.")
    unknown = [n for n in names if not (adaptive_root(cfg) / n).is_dir()]
    if unknown:
        raise SystemExit(f"Unknown setting(s): {unknown}. Available: {list_settings(cfg)}")

    complete = []
    for name in names:
        missing = missing_arrays(cfg, adaptive_root(cfg) / name, args.patients)
        if not missing:
            complete.append(name)
        elif explicit:
            raise SystemExit(f"Setting '{name}' is missing {len(missing)} array(s), e.g. {missing[0]}. "
                             f"Run: python scripts/02b_adaptive.py <same method/params> --patients "
                             f"{' '.join(args.patients)}")
        else:
            print(f"Skipping '{name}': {len(missing)} array(s) missing (e.g. {missing[0]}).")
    if not complete:
        raise SystemExit("No setting has arrays for all requested patients.")
    names = complete

    print(f"Stage A on {args.patients}: raw + {len(names)} setting(s), cap {args.cap} interictal windows/case")
    started = time.time()
    outcome = run_stage_a(cfg, {n: adaptive_root(cfg) / n for n in names}, args.patients,
                          interictal_cap=args.cap, seed=args.seed, m2_floor=args.m2_floor,
                          return_breakdown=args.breakdown)
    result, breakdown = outcome if args.breakdown else (outcome, None)
    out = Path(cfg["paths"]["tables"]) / "pilot_stage_a.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    shown = result[["setting", "m1_mean", "m1_ratio", "m2", "m2_ratio", "agreement", "proceeds"]]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(shown.sort_values("m1_ratio").to_string(index=False))
    if breakdown is not None:
        breakdown.to_csv(out.with_name("pilot_stage_a_by_feature.csv"))
        with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
            print("\nSeizure separability by feature type (mean 2|AUC-0.5|):")
            print(breakdown[["raw", *[c for c in breakdown.columns if c != "raw"]]].to_string())
    print(f"\n{int(result['proceeds'].sum())} setting(s) proceed; {time.time() - started:.0f} s. Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
