"""Tests for stage 3: the HeuDiConv heuristic, conversion plan, clean-up and validation."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import nibabel as nb
import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.data import get_testdata_file

from imaging_pipeline import bids, heuristic, validate
from imaging_pipeline.inventory import inventory


def series(description="", sequence="", dim1=64, derived=False, series_id="1-x"):
    """The SeqInfo fields the heuristic reads."""
    return SimpleNamespace(
        series_description=description,
        sequence_name=sequence,
        dim1=dim1,
        is_derived=derived,
        series_id=series_id,
    )


# --- heuristic -----------------------------------------------------------------


@pytest.mark.parametrize(
    "description, sequence, expected",
    [
        ("Ax DWI TENSOR R2", "epi2", heuristic.DWI),  # GE
        ("DT_HIGH_32DIR", "", heuristic.DWI),  # Philips
        ("dti_ax", "", heuristic.DWI),
        ("3D Ax T1 MP-RAGE_R22", "", heuristic.ANAT_T1W),
        ("Ax fMRI HB8 72sl int asc", "epiRT", heuristic.FUNC),  # GE
        ("epiRT IntAsc GD0", "epiRT", heuristic.FUNC),
        ("ax_asc_35sl", "*epfid2d1_64", heuristic.FUNC),  # Siemens, matched on sequence
        ("EPI_asc", "", heuristic.FUNC),  # Philips
    ],
)
def test_classify(description, sequence, expected):
    assert heuristic.classify(series(description, sequence)) == expected


def test_spin_echo_epi_with_direction_is_fieldmap():
    s = series("noPF_noPAT_noPOS_PEres100_ES0p59_BW2222_PA", "epse2d1_90")
    assert heuristic.classify(s) == "sub-{subject}/fmap/sub-{subject}_dir-PA_run-{item:02d}_epi"


@pytest.mark.parametrize(
    "s",
    [
        series("Phoenix Document", dim1=0),  # Siemens protocol report
        series("Ax fMRI", "epiRT", derived=True),
        series("localizer", "fl2d1"),  # nothing we convert
    ],
)
def test_classify_skips(s):
    assert heuristic.classify(s) is None


def test_infotodict_groups_series_by_key():
    info = heuristic.infotodict(
        [
            series("fMRI run a", series_id="2-a"),
            series("fMRI run b", series_id="3-b"),
            series("localizer", series_id="1-loc"),
        ]
    )
    assert info == {heuristic.create_key(heuristic.FUNC): ["2-a", "3-b"]}


# --- plan ----------------------------------------------------------------------


def inventory_rows(*rows):
    cols = ["study_uid", "series_uid", "manufacturer", "first_file"]
    return pd.DataFrame(rows, columns=cols)


def test_plan_one_subject_per_study_numbered_by_vendor(tmp_path):
    table = inventory_rows(
        ("s2", "a", "SIEMENS", "siemens/b/1.dcm"),
        ("g1", "b", "GE MEDICAL SYSTEMS", "ge/x/1.dcm"),
        ("s1", "c", "SIEMENS", "siemens/a/1.dcm"),
        ("s1", "d", "SIEMENS", "siemens/a2/1.dcm"),
        ("p1", "e", "Philips Medical Systems", "philips/x/1.dcm"),
    )
    studies = bids.plan(table, tmp_path).set_index("study_uid")
    assert studies.loc["s1", "subject"] == "siemens01"  # sorted by path within vendor
    assert studies.loc["s2", "subject"] == "siemens02"
    assert studies.loc["g1", "subject"] == "ge01"
    assert studies.loc["p1", "subject"] == "philips01"
    assert studies.loc["s1", "dirs"] == [str(tmp_path / "siemens/a"), str(tmp_path / "siemens/a2")]
    assert studies.loc["s1", "n_series"] == 2


def test_plan_unknown_vendor(tmp_path):
    table = inventory_rows(("x", "a", "TOSHIBA_MEC", "t/1.dcm"))
    assert bids.plan(table, tmp_path)["subject"].tolist() == ["other01"]


def test_plan_empty_inventory(tmp_path):
    assert bids.plan(inventory_rows(), tmp_path).empty


# --- tidy_subject ----------------------------------------------------------------


@pytest.fixture
def messy_subject(tmp_path: Path) -> Path:
    """A converted subject with the problems tidy_subject fixes."""
    func = tmp_path / "sub-x" / "func"
    dwi = tmp_path / "sub-x" / "dwi"
    func.mkdir(parents=True)
    dwi.mkdir()

    image = nb.Nifti1Image(np.zeros((2, 2, 2, 3), dtype=np.int16), np.eye(4))
    image.header.set_zooms((2, 2, 2, 1.0))
    bold = func / "sub-x_task-unknown_run-01_bold.nii.gz"
    nb.save(image, bold)
    bold.chmod(0o444)  # HeuDiConv outputs are read-only
    bold.with_suffix("").with_suffix(".json").write_text(json.dumps({"RepetitionTime": 1.35}))

    (dwi / "sub-x_run-01_dwi_heudiconv123_ADC.nii.gz").write_bytes(b"")
    (tmp_path / "sub-x" / "sub-x_scans.tsv").write_text(
        "filename\tacq_time\toperator\trandstr\nfunc/a.nii.gz\tn/a\tn/a\tabc\n"
    )
    return tmp_path


def test_tidy_fixes_header_tr_and_keeps_file_read_only(messy_subject):
    notes = bids.tidy_subject(messy_subject, "x")
    bold = messy_subject / "sub-x/func/sub-x_task-unknown_run-01_bold.nii.gz"
    assert nb.load(bold).header.get_zooms()[3] == pytest.approx(1.35)
    assert not bold.stat().st_mode & 0o200
    assert any("header TR 1s -> 1.35s" in n for n in notes)


def test_tidy_removes_extra_images(messy_subject):
    notes = bids.tidy_subject(messy_subject, "x")
    assert not list(messy_subject.rglob("*_heudiconv*"))
    assert "removed sub-x_run-01_dwi_heudiconv123_ADC.nii.gz" in notes


def test_tidy_drops_heudiconv_scans_columns(messy_subject):
    bids.tidy_subject(messy_subject, "x")
    scans = pd.read_csv(messy_subject / "sub-x/sub-x_scans.tsv", sep="\t")
    assert scans.columns.tolist() == ["filename", "acq_time"]


def test_tidy_leaves_matching_tr_alone(messy_subject):
    sidecar = messy_subject / "sub-x/func/sub-x_task-unknown_run-01_bold.json"
    sidecar.write_text(json.dumps({"RepetitionTime": 1.0}))
    notes = bids.tidy_subject(messy_subject, "x")
    assert not any("header TR" in n for n in notes)


# --- end to end -------------------------------------------------------------------


HAVE_HEUDICONV = shutil.which("heudiconv", path=bids._tool_env()["PATH"]) is not None


@pytest.mark.skipif(not HAVE_HEUDICONV, reason="heudiconv not installed")
def test_convert_end_to_end(tmp_path):
    """A one-file T1 study through inventory, HeuDiConv and dcm2niix to a BIDS dataset."""
    raw = tmp_path / "raw" / "t1"
    raw.mkdir(parents=True)
    ds = pydicom.dcmread(get_testdata_file("MR_small.dcm"))
    ds.SeriesDescription = "3D T1 MPRAGE"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M"]
    ds.save_as(raw / "1.dcm")

    out = tmp_path / "bids"
    result = bids.convert(inventory(tmp_path / "raw").series, tmp_path / "raw", out)

    assert result["status"].tolist() == ["converted"]
    assert (out / "sub-other01/anat/sub-other01_run-01_T1w.nii.gz").exists()
    assert json.loads((out / "dataset_description.json").read_text())["Name"] == bids.DATASET_NAME
    assert (out / "participants.tsv").read_text().split() == ["participant_id", "sub-other01"]


# --- validate ---------------------------------------------------------------------


def test_summarise_counts_by_severity_and_code():
    report = {
        "issues": {
            "issues": [
                {"severity": "error", "code": "A"},
                {"severity": "error", "code": "A"},
                {"severity": "warning", "code": "B"},
            ]
        }
    }
    assert validate.summarise(report) == {("error", "A"): 2, ("warning", "B"): 1}


def test_validator_prefers_local_install(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: f"/bin/{name}")
    assert validate.validator_command(tmp_path)[0] == "/bin/bids-validator"


def test_validator_falls_back_to_docker(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/docker" if name == "docker" else None)
    cmd = validate.validator_command(tmp_path)
    assert cmd[:2] == ["docker", "run"] and validate.DOCKER_IMAGE in cmd
