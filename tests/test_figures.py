"""Report figures: every function draws a non-empty PNG from small synthetic tables."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from eegpipe.evaluation import figures as fg

PATIENTS = [f"chb{i:02d}" for i in range(1, 6)]
KALMAN = "adaptive_kalman_qr3e-05_blk1"


def per_patient_tables():
    base = np.linspace(0.6, 0.95, 5)
    tables = {}
    for arm, shift in (("raw", 0.0), ("subject_standardised", 0.03), (KALMAN, 0.05)):
        for model in ("rf", "svm"):
            tables[(arm, model)] = pd.DataFrame(
                {"auc": base + shift}, index=pd.Index(PATIENTS, name="patient"))
    return tables


def assert_png(path):
    path = Path(path)
    assert path.exists() and path.stat().st_size > 1_000 and path.read_bytes()[:4] == b"\x89PNG"


def test_pretty_arm_names_and_colors():
    assert fg.pretty_arm("raw") == "Raw"
    assert fg.pretty_arm("raw_dev5") == "Raw"
    assert fg.pretty_arm("subject_standardised_dev5") == "Per-patient standardised"
    assert fg.pretty_arm("adaptive_kalman_qr3e-05_blk1_dev5") == "Kalman q/r 3e-05"
    assert fg.pretty_arm("adaptive_kalman_qr3e-05_blk1_standardised") == "Kalman q/r 3e-05 + standardised"
    assert fg.pretty_arm("adaptive_rls_L8_lam0.9999") == "RLS L8 λ=0.9999"
    assert fg.pretty_arm("adaptive_nlms_L16_mu0.02") == "NLMS L16 μ=0.02"
    assert fg.pretty_arm("something_else") == "something_else"
    assert fg.arm_color("adaptive_kalman_x") == fg.COLORS["kalman"]
    assert fg.arm_color("raw") == fg.COLORS["raw"]


def test_screen_and_convergence_figures(tmp_path):
    summary = pd.DataFrame({"method": ["nlms", "nlms", "rls"], "tag": ["nlms_a", "nlms_b", "rls_a"],
                            "worst_kept": [0.93, 0.80, 0.91], "passes": [True, False, True]})
    assert_png(fg.fig_screen(summary, tmp_path / "screen.png"))
    status = pd.DataFrame([
        {"setting": s, "case": c, "file": f"{c}_{n:02d}.edf", "status": "success",
         "rms_in": 40.0, "rms_out": 40.0 * (0.1 / (n + 1) + 0.02)}
        for s in ("nlms_a", "rls_a") for c in ("chb01", "chb21") for n in range(1, 5)])
    assert_png(fg.fig_convergence(status, tmp_path / "conv.png"))


def test_stage_a_figure_with_and_without_the_feature_table(tmp_path):
    stage_a = pd.DataFrame({"setting": ["raw", "adaptive_kalman_qr3e-05_blk1", "adaptive_rls_L8_lam0.9999"],
                            "m1_ratio": [1.0, 0.78, 0.40], "m2_ratio": [1.0, 0.97, 0.75]})
    by_feature = pd.DataFrame(np.random.default_rng(0).random((4, 3)),
                              index=["logvar", "linelength", "specentropy", "mobility"],
                              columns=stage_a["setting"])
    assert_png(fg.fig_stage_a(stage_a, by_feature, tmp_path / "a.png"))
    assert_png(fg.fig_stage_a(stage_a, None, tmp_path / "b.png"))


def test_per_patient_auc_forest_event_and_direction_figures(tmp_path):
    tables = per_patient_tables()
    assert_png(fg.fig_per_patient_auc(tables, ["raw", "subject_standardised", KALMAN], tmp_path / "auc.png", "t"))
    comparison = pd.DataFrame([
        {"model": m, "arm": KALMAN, "reference": "subject_standardised", "subset": s, "metric": "auc",
         "n_patients": 5, "mean_diff": d, "ci_low": d - 0.05, "ci_high": d + 0.06}
        for m in ("rf", "svm") for s, d in (("dev", 0.03), ("eval", -0.01))])
    assert_png(fg.fig_forest(comparison, tmp_path / "forest.png"))
    event = pd.DataFrame([
        {"arm": a, "model": m, "patient": p, "n_seizures": 3, "n_detected": 2, "fa_per_hour_event": 10.0 + i}
        for i, a in enumerate(("raw", "subject_standardised", KALMAN)) for m in ("rf", "svm") for p in PATIENTS[:3]])
    assert_png(fg.fig_event_tradeoff(event, tmp_path / "event.png"))
    direction = pd.DataFrame({"separability": [0.3, 0.2, 0.3, 0.25, 0.14], "agreement": [0.9, 0.95, 0.9, 0.85, 0.4]},
                             index=pd.Index(PATIENTS, name="patient"))
    assert_png(fg.fig_direction(direction, tmp_path / "dir.png"))


def test_script_makes_the_available_figures_and_skips_the_rest(tmp_cfg, tmp_path, write_cfg):
    cfg = tmp_cfg
    tables, logs = Path(cfg["paths"]["tables"]), Path(cfg["paths"]["logs"])
    tables.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    for (arm, model), table in per_patient_tables().items():
        table.reset_index().to_csv(tables / f"per_patient_{arm}__{model}.csv", index=False)
    pd.DataFrame({"method": ["nlms"], "tag": ["nlms_a"], "worst_kept": [0.9], "passes": [True]}).to_csv(
        tables / "adaptive_screen_summary.csv", index=False)
    pd.DataFrame({"patient": PATIENTS, "separability": 0.3, "agreement": 0.9}).to_csv(
        tables / "feature_direction_summary.csv", index=False)
    spec = importlib.util.spec_from_file_location(
        "make_figures", Path(__file__).resolve().parents[1] / "scripts" / "07_make_figures.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out_dir = tmp_path / "figs"
    cfg_path = write_cfg(cfg, tmp_path / "cfg.yaml")
    assert script.main(["--config", str(cfg_path), "--out-dir", str(out_dir)]) == 0
    made = {p.name for p in out_dir.glob("*.png")}
    assert {"screen.png", "direction.png", "eval10_auc.png"} <= made
    assert "stage_a.png" not in made and "event_tradeoff.png" not in made  # no input tables: skipped
