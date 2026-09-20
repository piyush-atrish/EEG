"""Self-tests of the synthetic fixtures (they are the foundation of B's and C's tests)."""

import mne
import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synth_signals import (
    make_synthetic_project,
    make_synthetic_summary_text,
)


@pytest.mark.parametrize("seed", range(12))
def test_project_guarantees(tmp_path, seed):
    out = make_synthetic_project(tmp_path, seed=seed)
    ann = pd.read_csv(out["annotations_csv"])
    idx = pd.read_csv(out["file_index_csv"])
    per_patient = ann.groupby("patient").size().reindex(out["patients"]).fillna(0)
    assert per_patient.between(1, 2).all()
    for _, grp in ann.groupby("file"):
        grp = grp.sort_values("seizure_start_s")
        assert list(grp["seizure_idx"]) == list(range(len(grp)))
        starts, ends = grp["seizure_start_s"].to_numpy(), grp["seizure_end_s"].to_numpy()
        assert (starts[1:] >= ends[:-1] + 15 - 1e-9).all()
        assert (ends > starts).all()
    firsts = idx.sort_values("file_order").groupby("case").head(1)
    assert (firsts["n_seizures"] == 0).all()
    assert idx.groupby("patient")["file_order"].apply(lambda s: s.is_unique).all()
    assert set(idx.loc[idx["case"] == "chb21", "patient"]) == {"chb01"}


def test_project_layout_and_contract(tmp_path):
    out = make_synthetic_project(tmp_path, seed=1)
    assert all(str(p).startswith(str(tmp_path)) for p in out["preprocessed"])
    x = np.load(out["preprocessed"][0])
    idx = pd.read_csv(out["file_index_csv"])
    assert x.dtype == np.float32 and x.shape == (18, int(idx["n_samples"].iloc[0]))
    assert list(idx.columns) == ["patient", "case", "file", "file_order", "duration_s",
                                 "n_samples", "n_seizures", "role", "include", "exclude_reason"]


def test_project_is_deterministic(tmp_path):
    a = make_synthetic_project(tmp_path / "a", seed=5)
    b = make_synthetic_project(tmp_path / "b", seed=5)
    assert pd.read_csv(a["annotations_csv"]).equals(pd.read_csv(b["annotations_csv"]))


def test_project_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError):
        make_synthetic_project(tmp_path, files_per_patient=1)


def test_summary_text_variants():
    assert "Seizure Start Time" in make_synthetic_summary_text("v1")
    assert "Seizure 2 Start Time" in make_synthetic_summary_text("v2")
    with pytest.raises(ValueError):
        make_synthetic_summary_text("v3")


def test_fake_chbmit_edfs_match_ground_truth(fake_chbmit):
    """Real EDF bytes: duplicate T8-P8, placeholder '-', shuffled order, microvolts."""
    mne.set_log_level("ERROR")
    channels = fake_chbmit["cfg"]["dataset"]["channels"]
    assert len(fake_chbmit["truth"]) == 9
    for edf, truth in fake_chbmit["truth"].items():
        raw = mne.io.read_raw_edf(edf, preload=False, verbose=False)
        names = raw.ch_names
        assert {"T8-P8-0", "T8-P8-1"} <= set(names)
        got = np.vstack(
            [raw.get_data(picks=[names.index("T8-P8-0" if c == "T8-P8" else c)])[0] * 1e6
             for c in channels]
        )
        assert np.abs(got - truth).max() < 0.02
