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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
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


def is_standardised_variant(arm: str) -> bool:
    """Adaptive arm with per-patient standardisation on top (drawn hatched, same colour as its base)."""
    return arm.startswith("adaptive_") and re.sub(r"_dev\d+$", "", arm).endswith("_standardised")


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


def convergence_settings(status: pd.DataFrame, cases: list[str], settings: list[str] | None = None) -> list[str]:
    """Settings drawn in the convergence figure: those with results for EVERY plotted case."""
    ok = status[status["status"] == "success"]
    per_case = [set(ok.loc[ok["case"] == case, "setting"]) for case in cases]
    common = set.intersection(*per_case) if per_case else set()
    chosen = sorted(common if settings is None else common & set(settings))
    return chosen


def fig_convergence(status: pd.DataFrame, out: str | Path, settings: list[str] | None = None,
                    cases: list[str] | None = None) -> Path:
    """Output/input RMS per file, in chronological order, one panel per case.

    A setting has the same colour in every panel, and only settings available for all plotted cases
    are drawn, so the legend is valid for every panel.
    """
    from eegpipe.adaptive.runner import file_order

    ok = status[status["status"] == "success"].copy()
    ok["ratio"] = ok["rms_out"] / ok["rms_in"]
    ok = ok.drop_duplicates(["setting", "file"], keep="last")
    cases = cases or sorted(ok["case"].unique())[:2]
    names = convergence_settings(ok, cases, settings)
    colors = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(names)}
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 3.8), sharey=True, squeeze=False)
    for ax, case in zip(axes[0], cases):
        for name in names:
            group = ok[(ok["case"] == case) & (ok["setting"] == name)]
            group = group.assign(order=group["file"].map(lambda f: file_order(f)[0])).sort_values("order")
            ax.plot(range(len(group)), group["ratio"], marker="o", color=colors[name], label=name)
        ax.set_title(case)
        ax.set_xlabel("File index (chronological)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0][0].set_ylabel("Output RMS / input RMS")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, max(1, len(labels))), fontsize=7)
    fig.suptitle("Filter convergence: a falling ratio means the filter is still converging")
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


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
    ax.margins(x=0.12, y=0.12)
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
    """Grouped bars of per-patient AUC (one panel per model) with a mean group at the end.

    Adaptive arms with standardisation on top are hatched in the colour of their base arm.
    """
    models = sorted({m for _, m in tables})
    fig, axes = plt.subplots(len(models), 1, figsize=(9, 3.2 * len(models) + 0.9), squeeze=False)
    handles = {}
    for ax, model in zip(axes[:, 0], models):
        matrix = per_patient_matrix(tables, model, "auc", arms)
        x = np.arange(len(matrix.index))
        width = 0.8 / len(matrix.columns)
        for i, arm in enumerate(matrix.columns):
            bars = ax.bar(x + i * width - 0.4 + width / 2, matrix[arm], width, color=arm_color(arm),
                          hatch="//" if is_standardised_variant(arm) else None,
                          edgecolor="white" if is_standardised_variant(arm) else None)
            handles.setdefault(arm, bars[0])
        ax.set_xticks(x, matrix.index)
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"AUC ({model.upper()})")
    fig.legend(list(handles.values()), [pretty_arm(a) for a in handles], loc="lower center",
               ncol=min(3, len(handles)), fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


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
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    markers = {"rf": "o", "svm": "s"}
    for row in data.itertuples():
        ax.scatter(row.fa, row.sens, color=arm_color(row.arm), marker=markers.get(row.model, "^"), s=80)
    arm_handles = [Line2D([], [], marker="o", ls="", color=arm_color(a), label=pretty_arm(a))
                   for a in dict.fromkeys(data["arm"])]
    model_handles = [Line2D([], [], marker=markers.get(m, "^"), ls="", color="k", label=m.upper())
                     for m in dict.fromkeys(data["model"])]
    ax.legend(handles=arm_handles + model_handles, fontsize=8, loc="lower right")
    ax.set_xlabel("False alarms per hour (mean over patients)")
    ax.set_ylabel("Fraction of seizures detected")
    ax.set_title("Event-level trade-off (upper left is better)")
    ax.set_ylim(0, 1.05)
    ax.margins(x=0.1)
    return _save(fig, out)


def place_labels(ax, xs, ys, texts, fontsize: int = 8, marker_radius: float = 8.0) -> None:
    """Annotate points choosing, for each label, the nearest position that touches no marker, no earlier
    label and stays inside the plot area.

    Candidates are tried from closest to farthest and scale with the label width, so a label sits right
    next to its own point. Distances are checked in display pixels (any axis scale). If every candidate
    collides, the first one is used.
    """
    ax.figure.canvas.draw()  # fixes the axis limits so data -> pixel is final
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    px_per_pt = ax.figure.dpi / 72.0
    height_pt = 1.3 * fontsize
    area = ax.bbox
    placed: list[tuple[float, float, float, float]] = []

    def overlaps(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    markers = [(x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius) for x, y in pts]
    for i, (text, (px, py)) in enumerate(zip(texts, pts)):
        width_pt = 0.62 * fontsize * len(text)
        candidates = [
            (6, 3), (6, -height_pt - 1), (-width_pt - 6, 3), (-width_pt - 6, -height_pt - 1),
            (-width_pt / 2, 8), (-width_pt / 2, -height_pt - 7),
            (10, 12), (-width_pt - 10, 12), (10, -height_pt - 11), (-width_pt - 10, -height_pt - 11),
        ]
        chosen = None
        for dx, dy in candidates:
            x0, y0 = px + dx * px_per_pt, py + dy * px_per_pt
            box = (x0, y0, x0 + width_pt * px_per_pt, y0 + height_pt * px_per_pt)
            inside = box[0] >= area.x0 and box[2] <= area.x1 and box[1] >= area.y0 and box[3] <= area.y1
            if inside and not any(overlaps(box, m) for m in markers) and not any(overlaps(box, o) for o in placed):
                chosen = (dx, dy, box)
                break
        if chosen is None:
            dx, dy = candidates[0]
            x0, y0 = px + dx * px_per_pt, py + dy * px_per_pt
            chosen = (dx, dy, (x0, y0, x0 + width_pt * px_per_pt, y0 + height_pt * px_per_pt))
        placed.append(chosen[2])
        ax.annotate(text, (xs[i], ys[i]), fontsize=fontsize, xytext=chosen[:2], textcoords="offset points")


def fig_direction(summary: pd.DataFrame, out: str | Path) -> Path:
    """Direction agreement vs separability per patient; the lowest-agreement patient is highlighted."""
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    worst = summary["agreement"].idxmin() if len(summary) else None
    xs, ys = summary["agreement"].to_numpy(dtype=float), summary["separability"].to_numpy(dtype=float)
    colors = ["#d62728" if patient == worst else "#1f77b4" for patient in summary.index]
    ax.scatter(xs, ys, c=colors, s=60)
    ax.margins(x=0.1, y=0.1)
    place_labels(ax, xs, ys, [str(p) for p in summary.index])
    ax.set_xlabel("Direction agreement with the other patients")
    ax.set_ylabel("Mean seizure separability of the features")
    ax.set_title("Do seizures move the features the same way?", fontsize=10)
    return _save(fig, out)