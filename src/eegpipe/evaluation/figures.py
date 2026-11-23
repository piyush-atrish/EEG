"""Report figures built only from the saved result tables (chunk 8.1).

Every function takes plain DataFrames and an output path, draws one figure, and returns the path, so
the whole set can be regenerated with ``python scripts/07_make_figures.py`` after any re-run.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eegpipe.evaluation.compare import per_patient_matrix  # noqa: E402

COLORS = {"raw": "#8c8c8c", "std": "#1f77b4", "kalman": "#d95f02", "nlms": "#7570b3", "rls": "#1b9e77"}
DPI = 150


def pretty_arm(arm: str) -> str:
    """Readable legend name for an arm label such as ``adaptive_kalman_qr3e-05_blk1_dev5``."""
    name = re.sub(r"_dev\d+$", "", arm)
    suffix = ""
    if name.endswith("_standardised") and name.startswith("adaptive_"):
        name, suffix = name[: -len("_standardised")], " + standardised"
    if name == "raw":
        return "Raw"
    if name.startswith("subject_standardised"):
        return "Per-patient standardised"
    if match := re.fullmatch(r"adaptive_kalman_qr([0-9.e+-]+)_blk[0-9.]+", name):
        return f"Kalman q/r {match.group(1)}{suffix}"
    if match := re.fullmatch(r"adaptive_rls_L(\d+)_lam([0-9.]+)", name):
        return f"RLS L{match.group(1)} λ={match.group(2)}{suffix}"
    if match := re.fullmatch(r"adaptive_nlms_L(\d+)_mu([0-9.]+)", name):
        return f"NLMS L{match.group(1)} μ={match.group(2)}{suffix}"
    return arm


def arm_color(arm: str) -> str:
    low = arm.lower()
    for key, color in (("kalman", COLORS["kalman"]), ("rls", COLORS["rls"]), ("nlms", COLORS["nlms"])):
        if key in low:
            return color
    return COLORS["std"] if "standardised" in low else COLORS["raw"]


def _save(fig, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig_screen(summary: pd.DataFrame, out: str | Path, threshold: float = 0.85) -> Path:
    """Synthetic seizure-preservation screen: worst-case separability kept, per setting."""
    data = summary.sort_values(["method", "worst_kept"], ascending=[True, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(data) + 1.2))
    colors = ["#2ca02c" if ok else "#d62728" for ok in data["passes"]]
    ax.barh(data["tag"], data["worst_kept"], color=colors)
    ax.axvline(threshold, color="k", ls="--", lw=1)
    ax.set_xlabel("Worst-case fraction of burst separability kept")
    ax.set_title("Synthetic seizure-preservation screen (green = passes)")
    ax.set_xlim(0, 1.05)
    return _save(fig, out)


def fig_convergence(status: pd.DataFrame, out: str | Path, settings: list[str] | None = None,
                    cases: list[str] | None = None) -> Path:
    """Output/input RMS per file, in chronological order, one panel per case."""
    from eegpipe.adaptive.runner import file_order

    ok = status[status["status"] == "success"].copy()
    ok["ratio"] = ok["rms_out"] / ok["rms_in"]
    ok = ok.drop_duplicates(["setting", "file"], keep="last")
    if settings:
        ok = ok[ok["setting"].isin(settings)]
    cases = cases or sorted(ok["case"].unique())[:2]
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 3.6), sharey=True, squeeze=False)
    for ax, case in zip(axes[0], cases):
        part = ok[ok["case"] == case]
        for setting, group in part.groupby("setting"):
            group = group.assign(order=group["file"].map(lambda f: file_order(f)[0])).sort_values("order")
            ax.plot(range(len(group)), group["ratio"], marker="o", label=setting)
        ax.set_title(case)
        ax.set_xlabel("File index (chronological)")
    axes[0][0].set_ylabel("Output RMS / input RMS")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Filter convergence: a falling ratio means the filter is still converging")
    return _save(fig, out)


def fig_stage_a(stage_a: pd.DataFrame, by_feature: pd.DataFrame | None, out: str | Path,
                m2_floor: float = 0.9) -> Path:
    """Pilot stage A: patient variance (M1) vs seizure separability (M2), plus the per-feature-type table."""
    ncols = 2 if by_feature is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.4), squeeze=False)
    ax = axes[0][0]
    for row in stage_a.itertuples():
        ax.scatter(row.m1_ratio, row.m2_ratio, color=arm_color(row.setting), s=60)
        ax.annotate(pretty_arm(row.setting), (row.m1_ratio, row.m2_ratio), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(m2_floor, color="k", ls="--", lw=1)
    ax.axvline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("M1 ratio (patient variance vs raw; lower is better)")
    ax.set_ylabel("M2 ratio (seizure separability vs raw)")
    ax.set_title("Pilot stage A (development patients)")
    if by_feature is not None:
        ax2 = axes[0][1]
        im = ax2.imshow(by_feature.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax2.set_yticks(range(len(by_feature.index)), by_feature.index, fontsize=7)
        ax2.set_xticks(range(len(by_feature.columns)), [pretty_arm(c) for c in by_feature.columns],
                       rotation=45, ha="right", fontsize=7)
        ax2.set_title("Seizure separability by feature type")
        fig.colorbar(im, ax=ax2)
    return _save(fig, out)


def fig_per_patient_auc(tables: dict, arms: list[str], out: str | Path, title: str = "") -> Path:
    """Grouped bars of per-patient AUC (one panel per model) with a mean group at the end."""
    models = sorted({m for _, m in tables})
    fig, axes = plt.subplots(len(models), 1, figsize=(9, 3.2 * len(models)), squeeze=False)
    for ax, model in zip(axes[:, 0], models):
        matrix = per_patient_matrix(tables, model, "auc", arms)
        x = np.arange(len(matrix.index))
        width = 0.8 / len(matrix.columns)
        for i, arm in enumerate(matrix.columns):
            ax.bar(x + i * width - 0.4 + width / 2, matrix[arm], width, label=pretty_arm(arm),
                   color=arm_color(arm))
        ax.set_xticks(x, matrix.index)
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"AUC ({model.upper()})")
    axes[0, 0].legend(fontsize=7, ncol=2, loc="lower left")
    if title:
        fig.suptitle(title)
    return _save(fig, out)


def fig_forest(comparison: pd.DataFrame, out: str | Path, metric: str = "auc",
               subsets: tuple[str, ...] = ("dev", "eval")) -> Path:
    """Paired differences (arm minus reference) with bootstrap intervals, per model and subset."""
    data = comparison[(comparison["metric"] == metric) & comparison["subset"].isin(subsets)].copy()
    data["label"] = data["arm"].map(pretty_arm) + " [" + data["subset"] + ", n=" + data["n_patients"].astype(str) + "]"
    models = sorted(data["model"].unique())
    reference = pretty_arm(str(data["reference"].iloc[0])) if len(data) else ""
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 0.45 * (len(data) / max(1, len(models))) + 1.8),
                             squeeze=False, sharex=True)
    for ax, model in zip(axes[0], models):
        part = data[data["model"] == model].reset_index(drop=True)
        y = np.arange(len(part))
        ax.errorbar(part["mean_diff"], y, xerr=[part["mean_diff"] - part["ci_low"], part["ci_high"] - part["mean_diff"]],
                    fmt="o", color="k", capsize=3)
        ax.axvline(0, color="r", lw=1)
        ax.set_yticks(y, part["label"], fontsize=7)
        ax.set_title(model.upper())
        ax.set_xlabel(f"{metric.upper()} difference vs {reference}")
    return _save(fig, out)


def fig_event_tradeoff(event: pd.DataFrame, out: str | Path) -> Path:
    """Seizure detection rate vs false alarms per hour for each arm and model."""
    rows = []
    for (arm, model), g in event.groupby(["arm", "model"]):
        rows.append({"arm": arm, "model": model,
                     "sens": g["n_detected"].sum() / g["n_seizures"].sum(),
                     "fa": g["fa_per_hour_event"].mean()})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for row in data.itertuples():
        ax.scatter(row.fa, row.sens, color=arm_color(row.arm), marker="o" if row.model == "rf" else "s", s=70)
        ax.annotate(f"{pretty_arm(row.arm)} ({row.model})", (row.fa, row.sens), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("False alarms per hour (mean over patients)")
    ax.set_ylabel("Fraction of seizures detected")
    ax.set_title("Event-level trade-off (circles: RF, squares: SVM)")
    ax.set_ylim(0, 1.05)
    return _save(fig, out)


def fig_direction(summary: pd.DataFrame, out: str | Path) -> Path:
    """Direction agreement vs separability per patient; the lowest-agreement patient is highlighted."""
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    worst = summary["agreement"].idxmin() if len(summary) else None
    for patient, row in summary.iterrows():
        ax.scatter(row["agreement"], row["separability"], color="#d62728" if patient == worst else "#1f77b4", s=60)
        ax.annotate(str(patient), (row["agreement"], row["separability"]), fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Direction agreement with the other patients")
    ax.set_ylabel("Mean seizure separability of the features")
    ax.set_title("Do seizures move the features the same way in every patient?")
    return _save(fig, out)
