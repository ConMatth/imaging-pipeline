"""Tests for stage 4: metadata scrubbing, defacing and the leftover-identifier audit."""

import json
from importlib.resources import files
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

from imaging_pipeline import deidentify as deid

# --- metadata ------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "PatientName", "PatientID", "PatientBirthDate", "InstitutionName", "InstitutionAddress",
        "InstitutionalDepartmentName", "StationName", "DeviceSerialNumber", "AcquisitionDateTime",
        "StudyInstanceUID", "SeriesInstanceUID", "AccessionNumber", "ImageComments",
        "ReferringPhysicianName", "OperatorsName", "StudyDescription", "ProcedureStepDescription",
    ],
)  # fmt: skip
def test_identifying_keys(key):
    assert deid.is_identifying_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "PatientPosition", "ImageOrientationPatientDICOM", "SeriesDescription", "ProtocolName",
        "AcquisitionTime", "RepetitionTime", "Manufacturer", "TaskName", "SequenceName",
    ],
)  # fmt: skip
def test_scan_keys_are_kept(key):
    assert not deid.is_identifying_key(key)


def test_scrub_sidecar(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"InstitutionName": "orange", "EchoTime": 0.03, "StationName": "1"}))
    assert deid.scrub_sidecar(path) == ["InstitutionName", "StationName"]
    assert json.loads(path.read_text()) == {"EchoTime": 0.03}


def test_scrub_scans_tsv_keeps_time_of_day(tmp_path):
    path = tmp_path / "sub-x_scans.tsv"
    path.write_text("filename\tacq_time\na.nii.gz\t2020-09-29T00:14:57\nb.nii.gz\tn/a\n")
    assert deid.scrub_scans_tsv(path) == 1
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert table["acq_time"].tolist() == ["1925-01-01T00:14:57", "n/a"]


def nifti(path: Path, shape=(4, 4, 4), descrip=b"TE=3.2;Time=2056.000") -> Path:
    image = nb.Nifti1Image(np.ones(shape, dtype=np.int16), np.eye(4))
    image.header["descrip"] = descrip
    nb.save(image, path)
    return path


def test_scrub_nifti_header(tmp_path):
    path = nifti(tmp_path / "a.nii.gz")
    assert deid.scrub_nifti_header(path)
    assert nb.load(path).header["descrip"].item() == b""
    assert nb.load(path).get_data_dtype() == np.int16
    assert not deid.scrub_nifti_header(path)  # nothing left to change


# --- defacing ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def moved_head(tmp_path_factory):
    """pydeface's head template at 3 mm, rotated and shifted in space.

    Only the affine changes, so the true face mask in voxel space is the
    template's own face mask: a registration that recovers the pose matches it.
    """
    data_dir = files("pydeface") / "data"
    template = nb.load(str(data_dir / "mean_reg2mean.nii.gz"))
    step = 3
    head = np.asarray(template.dataobj)[::step, ::step, ::step].astype(np.float32)
    face = np.asarray(nb.load(str(data_dir / "facemask.nii.gz")).dataobj)[::step, ::step, ::step]

    angle = np.deg2rad(10)
    c, s = np.cos(angle), np.sin(angle)
    pose = np.array([[1, 0, 0, 6], [0, c, -s, -9], [0, s, c, 12], [0, 0, 0, 1]])
    affine = pose @ template.affine @ np.diag([step, step, step, 1])

    path = tmp_path_factory.mktemp("head") / "sub-x_T1w.nii.gz"
    nb.save(nb.Nifti1Image(head, affine), path)
    return path, face < 0.5


def test_face_mask_finds_the_face(moved_head):
    path, true_face = moved_head
    removed = deid.face_mask(path) < 0.5
    dice = 2 * (removed & true_face).sum() / (removed.sum() + true_face.sum())
    assert dice > 0.9


def test_deface_blanks_face_and_saves_picture(moved_head, tmp_path):
    source, true_face = moved_head
    path = tmp_path / source.name
    path.write_bytes(source.read_bytes())
    png = tmp_path / "qc" / "sub-x_T1w.png"

    fraction = deid.deface(path, png)

    defaced = nb.load(path).get_fdata()
    face_left = np.count_nonzero(defaced[true_face]) / np.count_nonzero(true_face)
    assert face_left < 0.02  # only voxels right on the mask's edge can differ
    assert 0 < fraction < 0.5
    assert png.stat().st_size > 0


def test_deface_rejects_4d(tmp_path):
    with pytest.raises(ValueError, match="3D"):
        deid.deface(nifti(tmp_path / "bold.nii.gz", shape=(4, 4, 4, 2)))


# --- whole dataset -----------------------------------------------------------------


@pytest.fixture
def bids_dataset(tmp_path) -> Path:
    """A small identifiable dataset, like stage 3 writes (no anatomical image)."""
    root = tmp_path / "bids"
    func = root / "sub-x" / "func"
    func.mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "t", "BIDSVersion": "1"}))
    (root / ".heudiconv" / "x" / "info").mkdir(parents=True)
    (root / ".heudiconv" / "x" / "info" / "dicominfo.tsv").write_text("patient_id\nJOHN\n")
    bold = nifti(func / "sub-x_task-a_bold.nii.gz")
    bold.chmod(0o444)  # HeuDiConv outputs are read-only
    (func / "sub-x_task-a_bold.json").write_text(
        json.dumps({"RepetitionTime": 2, "InstitutionName": "Hospital", "PatientPosition": "HFS"})
    )
    (root / "sub-x" / "sub-x_scans.tsv").write_text(
        "filename\tacq_time\nfunc/sub-x_task-a_bold.nii.gz\t2020-01-02T10:00:00\n"
    )
    return root


def test_audit_finds_identifiers(bids_dataset):
    problems = deid.find_identifiers(bids_dataset)
    assert any("HeuDiConv" in p for p in problems)
    assert any("identifying keys ['InstitutionName']" in p for p in problems)
    assert any("real scan dates" in p for p in problems)
    assert any("NIfTI header" in p for p in problems)


def test_deidentify_dataset_leaves_nothing_identifying(bids_dataset, tmp_path):
    out = tmp_path / "deid"
    changes = deid.deidentify(bids_dataset, out)

    assert deid.find_identifiers(out) == []
    assert not (out / ".heudiconv").exists()
    sidecar = json.loads((out / "sub-x/func/sub-x_task-a_bold.json").read_text())
    assert sidecar == {"RepetitionTime": 2, "PatientPosition": "HFS"}
    assert len(changes) == 4  # .heudiconv, sidecar, scans.tsv, NIfTI header
    assert deid.find_identifiers(bids_dataset) != []  # source untouched


def test_deidentify_replaces_previous_output(bids_dataset, tmp_path):
    out = tmp_path / "deid"
    deid.deidentify(bids_dataset, out)
    (out / "stale.txt").write_text("from an old run")
    deid.deidentify(bids_dataset, out)
    assert not (out / "stale.txt").exists()


def test_deidentify_wont_delete_unrelated_folder(bids_dataset, tmp_path):
    out = tmp_path / "precious"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")
    with pytest.raises(FileExistsError):
        deid.deidentify(bids_dataset, out)
    assert (out / "notes.txt").exists()


def test_deidentify_wont_write_inside_source(bids_dataset):
    with pytest.raises(ValueError):
        deid.deidentify(bids_dataset, bids_dataset / "derivatives" / "deid")


# --- skull-stripped check ----------------------------------------------------------


def ball(radius_mm, shape=(100, 110, 100), voxel_mm=2.0, background="zero", centre=None):
    """A bright ball (stand-in for a brain or head) in a grid of 2 mm voxels."""
    grid = np.indices(shape).astype(float)
    centre = np.array(centre if centre is not None else [s / 2 for s in shape])
    distance = np.sqrt(sum((g - c) ** 2 for g, c in zip(grid, centre, strict=True))) * voxel_mm
    data = np.where(distance <= radius_mm, 500.0, 0.0)
    if background == "noise":
        data += np.random.default_rng(0).uniform(1, 20, shape)
    return nb.Nifti1Image(data.astype(np.float32), np.diag([voxel_mm] * 3 + [1]))


def check(tmp_path, image, sidecar=None):
    path = tmp_path / "sub-x_T1w.nii.gz"
    nb.save(image, path)
    if sidecar is not None:
        (tmp_path / "sub-x_T1w.json").write_text(json.dumps(sidecar))
    return deid.check_skull_stripped(path)


def test_brain_on_zero_background_is_stripped(tmp_path):
    stripped, reason = check(tmp_path, ball(70))  # ~1.4 L
    assert stripped
    assert "L of tissue" in reason


def test_head_on_noise_background_is_not_stripped(tmp_path):
    assert not check(tmp_path, ball(70, background="noise"))[0]


def test_large_head_on_zero_background_is_not_stripped(tmp_path):
    assert not check(tmp_path, ball(95))[0]  # ~3.6 L: a whole head


def test_cropped_head_on_zero_background_is_not_stripped(tmp_path):
    """A head scan with a narrow field of view: small volume, but it runs off the edge."""
    image = ball(70, shape=(100, 110, 40), centre=[50, 55, 10])
    stripped, reason = check(tmp_path, image)
    assert not stripped
    assert "tissue at image edges" in reason


def test_head_template_is_not_stripped(moved_head):
    assert not deid.check_skull_stripped(moved_head[0])[0]


@pytest.mark.parametrize("value", [True, False])
def test_sidecar_overrides_measurement(tmp_path, value):
    image = ball(95) if value else ball(70)  # measurement would say the opposite
    assert check(tmp_path, image, {"SkullStripped": value}) == (
        value,
        f"sidecar says SkullStripped={str(value).lower()}",
    )


def test_deidentify_leaves_stripped_image_alone(bids_dataset, tmp_path, monkeypatch):
    anat = bids_dataset / "sub-x" / "anat"
    anat.mkdir()
    nb.save(ball(70), anat / "sub-x_T1w.nii.gz")

    def no_deface(*args, **kwargs):
        raise AssertionError("a skull-stripped image must not be defaced")

    monkeypatch.setattr(deid, "deface", no_deface)
    qc = tmp_path / "qc"
    changes = deid.deidentify(bids_dataset, tmp_path / "deid", qc)

    row = changes[changes["file"] == "sub-x/anat/sub-x_T1w.nii.gz"]
    assert row["change"].item().startswith("not defaced: already skull-stripped")
    assert (qc / "sub-x_T1w.png").exists()
    assert deid.find_identifiers(tmp_path / "deid") == []
