"""Compare adaptive settings by output/input RMS per file (convergence check).

For a well-converged whitening filter the ratio settles to a stable small value; a ratio that keeps
falling from file to file means the filter is still converging. Reads results/logs/adaptive_status.csv
(latest successful row per setting and file).

Usage (from the repo root):
    python tools/adaptive_ratio.py
    python tools/adaptive_ratio.py --cases chb01 chb21
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eegpipe.adaptive.runner import file_order, status_csv
from eegpipe.config import load_config


def ratio_table(log: pd.DataFrame, cases: list[str] | None = None) -> pd.DataFrame:
    """Files x settings table of rms_out / rms_in (latest successful run of each file)."""
    ok = log[log["status"] == "success"].copy()
    if cases:
        ok = ok[ok["case"].isin(cases)]
    ok["ratio"] = ok["rms_out"] / ok["rms_in"]
    latest = ok.drop_duplicates(["setting", "file"], keep="last")
    table = latest.pivot(index="file", columns="setting", values="ratio")
    # group by case first (chb01_*, then chb21_*), then chronological within the case
    return table.loc[sorted(table.index, key=lambda f: (f.split("_")[0], file_order(f)))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cases", nargs="+", default=None)
    args = parser.parse_args(argv)
    path = status_csv(load_config(args.config))
    if not Path(path).exists():
        raise SystemExit(f"No status log at {path}. Run scripts/02b_adaptive.py first.")
    table = ratio_table(pd.read_csv(path), args.cases)
    if table.empty:
        raise SystemExit("No successful rows in the log.")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
