"""Tests for the DICOM inventory, using small example files bundled with pydicom."""

import shutil
from pathlib import Path

import pandas as pd
import pytest
from pydicom.data import get_testdata_file

from imaging_pipeline.inventory import inventory, main

MR = Path(get_testdata_file("MR_small.dcm"))
CT = Path(get_testdata_file("CT_small.dcm"))


@pytest.fixture
def dicom_dir(tmp_path: Path) -> Path:
    """One MR series split over two files, one CT file, and a stray text file."""
    (tmp_path / "mr").mkdir()
    shutil.copy(MR, tmp_path / "mr" / "img1.dcm")
    shutil.copy(MR, tmp_path / "mr" / "img2.dcm")
    shutil.copy(CT, tmp_path / "ct.dcm")
    (tmp_path / "notes.txt").write_text("not a DICOM file")
    return tmp_path


def test_one_row_per_series(dicom_dir):
    result = inventory(dicom_dir)
    assert len(result.series) == 2
    assert sorted(result.series["modality"]) == ["CT", "MR"]


def test_counts_files_per_series(dicom_dir):
    series = inventory(dicom_dir).series.set_index("modality")
    assert series.loc["MR", "n_files"] == 2
    assert series.loc["CT", "n_files"] == 1


def test_skips_non_dicom_files(dicom_dir):
    result = inventory(dicom_dir)
    assert [p.name for p in result.skipped] == ["notes.txt"]


def test_numeric_fields_are_numbers(dicom_dir):
    series = inventory(dicom_dir).series
    assert pd.api.types.is_numeric_dtype(series["rows"])
    assert (series["rows"] > 0).all()


def test_empty_folder_gives_empty_table(tmp_path):
    result = inventory(tmp_path)
    assert result.series.empty
    assert "series_uid" in result.series.columns


def test_missing_folder_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        inventory(tmp_path / "nope")


def test_cli_writes_csv(dicom_dir, tmp_path):
    out = tmp_path / "out" / "inventory.csv"
    assert main([str(dicom_dir), "--out", str(out)]) == 0
    assert out.read_text().count("\n") == 3  # header + 2 series
