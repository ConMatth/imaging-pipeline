"""Convert raw DICOM to a BIDS dataset with HeuDiConv and dcm2niix.

The inventory from stage 2 is the plan: every DICOM study becomes one BIDS
subject, and HeuDiConv is run once per study on just that study's series
folders. Running per study means one unreadable study is recorded as failed
instead of stopping the whole conversion.

Subject labels are made up here (``siemens01``, ``ge02``...), never taken from
patient IDs. The table written to ``--subjects`` links each label back to its
study UID, so treat it as a re-identification key and keep it out of shared data.

Usage:
    python -m imaging_pipeline.bids outputs/inventory.csv --raw data/raw \\
        --bids data/bids --subjects outputs/bids_subjects.csv
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd

HEURISTIC = Path(__file__).with_name("heuristic.py")

DATASET_NAME = "dcm_qa multi-vendor DICOM test data"
SOURCES = ("dcm_qa", "dcm_qa_ge", "dcm_qa_philips")
README = """\
Multi-vendor MRI test series (Siemens, GE, Philips) from the public dcm_qa
repositories by Chris Rorden: https://github.com/neurolabusc

Converted by imaging-pipeline with HeuDiConv and dcm2niix. These are scanner
test acquisitions, not a study: each DICOM study is one subject, and functional
runs use the placeholder task label "unknown".
"""

_VENDORS = {"siemens": "siemens", "ge": "ge", "philips": "philips"}


def _vendor(manufacturer) -> str:
    first_word = str(manufacturer or "").split(" ")[0].lower()
    return _VENDORS.get(first_word, "other")


def plan(inventory: pd.DataFrame, raw_root: str | Path) -> pd.DataFrame:
    """One row per study: its subject label and the series folders that belong to it."""
    raw_root = Path(raw_root)
    if inventory.empty:
        return pd.DataFrame(columns=["subject", "study_uid", "manufacturer", "n_series", "dirs"])

    table = inventory.assign(
        vendor=inventory["manufacturer"].map(_vendor),
        dir=inventory["first_file"].map(lambda f: str(raw_root / Path(f).parent)),
    )
    studies = (
        table.groupby("study_uid", sort=False)
        .agg(
            vendor=("vendor", "first"),
            manufacturer=("manufacturer", "first"),
            n_series=("series_uid", "size"),
            first_file=("first_file", "min"),
            dirs=("dir", lambda d: sorted(set(d))),
        )
        .reset_index()
        .sort_values(["vendor", "first_file"])
    )
    number = studies.groupby("vendor").cumcount() + 1
    studies["subject"] = studies["vendor"] + number.map("{:02d}".format)
    columns = ["subject", "study_uid", "manufacturer", "n_series", "dirs"]
    return studies[columns].reset_index(drop=True)


def _tool_env() -> dict[str, str]:
    """Environment with this Python's bin folder first, so heudiconv finds the venv's dcm2niix."""
    env = os.environ.copy()
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    return env


def _heudiconv() -> str:
    exe = shutil.which("heudiconv", path=_tool_env()["PATH"])
    if exe is None:
        raise FileNotFoundError("heudiconv not found; install with: pip install -e .")
    return exe


def convert_study(subject: str, dirs: list[str], bids_dir: Path) -> subprocess.CompletedProcess:
    """Run HeuDiConv on one study's series folders."""
    cmd = [
        _heudiconv(),
        "--files", *dirs,
        "--subjects", subject,
        "--grouping", "all",  # the folders are already one study
        "--heuristic", str(HEURISTIC),
        "--converter", "dcm2niix",
        "--bids", "notop",  # top-level files are written once, by finish_dataset
        "--minmeta",
        "--overwrite",
        "--outdir", str(bids_dir),
    ]  # fmt: skip
    return subprocess.run(cmd, capture_output=True, text=True, env=_tool_env())


def tidy_subject(bids_dir: Path, subject: str) -> list[str]:
    """Fix known conversion problems in one subject's folder. Returns a note per change.

    - dcm2niix writes extra images some scanners bundle into a series (e.g. a
      Philips ADC map in a DWI series) as ``*_heudiconvNNN_*`` files, which
      aren't valid BIDS. They're derived data, so they're removed.
    - When a series has a gap between volumes, dcm2niix puts the true volume
      interval in the sidecar but leaves the NIfTI header at the acquisition
      time. The header is set to match the sidecar.
    - HeuDiConv's ``operator`` and ``randstr`` columns in scans.tsv are dropped.
    """
    subject_dir = bids_dir / f"sub-{subject}"
    notes = []

    for extra in sorted(subject_dir.rglob("*_heudiconv[0-9]*")):
        extra.unlink()
        notes.append(f"removed {extra.name}")

    for sidecar in sorted(subject_dir.rglob("*_bold.json")):
        image_path = sidecar.with_suffix(".nii.gz")
        sidecar_tr = json.loads(sidecar.read_text()).get("RepetitionTime")
        if sidecar_tr is None or not image_path.exists():
            continue
        image = nb.load(image_path)
        zooms = image.header.get_zooms()
        if len(zooms) > 3 and abs(zooms[3] - sidecar_tr) > 1e-3:
            header = image.header.copy()
            header.set_zooms((*zooms[:3], sidecar_tr, *zooms[4:]))
            fixed = nb.Nifti1Image(np.asanyarray(image.dataobj), image.affine, header)
            mode = image_path.stat().st_mode  # HeuDiConv leaves its outputs read-only
            image_path.chmod(mode | stat.S_IWUSR)
            nb.save(fixed, image_path)
            image_path.chmod(mode)
            notes.append(f"set {image_path.name} header TR {zooms[3]:g}s -> {sidecar_tr:g}s")

    for scans in subject_dir.glob("*_scans.tsv"):
        table = pd.read_csv(scans, sep="\t")
        table.drop(columns=["operator", "randstr"], errors="ignore").to_csv(
            scans, sep="\t", index=False
        )
    return notes


def _last_error(stderr: str) -> str:
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    return lines[-1][:300] if lines else "heudiconv failed with no output"


def finish_dataset(bids_dir: Path, subjects: list[str]) -> None:
    """Write the dataset-level files: description, README, participants and ignore list."""
    description = {
        "Name": DATASET_NAME,
        "BIDSVersion": "1.10.0",
        "DatasetType": "raw",
        "License": "BSD-2-Clause",
        "Authors": ["Chris Rorden (source DICOM)"],
        "SourceDatasets": [{"URL": f"https://github.com/neurolabusc/{repo}"} for repo in SOURCES],
        "GeneratedBy": [{"Name": "heudiconv"}, {"Name": "imaging-pipeline"}],
    }
    (bids_dir / "dataset_description.json").write_text(json.dumps(description, indent=2) + "\n")
    (bids_dir / "README").write_text(README)
    pd.DataFrame({"participant_id": [f"sub-{s}" for s in sorted(subjects)]}).to_csv(
        bids_dir / "participants.tsv", sep="\t", index=False
    )
    (bids_dir / ".bidsignore").write_text(".heudiconv/\n")


def convert(inventory: pd.DataFrame, raw_root: str | Path, bids_dir: str | Path) -> pd.DataFrame:
    """Convert every study in the inventory. Returns the plan with a status per study."""
    bids_dir = Path(bids_dir)
    bids_dir.mkdir(parents=True, exist_ok=True)
    studies = plan(inventory, raw_root)

    statuses, notes = [], []
    for study in studies.itertuples():
        result = convert_study(study.subject, study.dirs, bids_dir)
        produced = any((bids_dir / f"sub-{study.subject}").rglob("*.nii.gz"))
        if result.returncode != 0:
            status, note = "failed", _last_error(result.stderr)
        elif not produced:
            status, note = "no_bids_series", ""  # every series was skipped by the heuristic
        else:
            status, note = "converted", "; ".join(tidy_subject(bids_dir, study.subject))
        statuses.append(status)
        notes.append(note)
        print(f"sub-{study.subject}: {status} {note}".rstrip())

    converted = [s for s, st in zip(studies["subject"], statuses, strict=True) if st == "converted"]
    finish_dataset(bids_dir, converted)
    return studies.assign(status=statuses, notes=notes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert inventoried DICOM to BIDS.")
    parser.add_argument("inventory", type=Path, help="Inventory CSV from stage 2")
    parser.add_argument("--raw", type=Path, required=True, help="Folder the inventory scanned")
    parser.add_argument("--bids", type=Path, required=True, help="BIDS dataset to write")
    parser.add_argument("--subjects", type=Path, help="CSV of subject label, study and status")
    args = parser.parse_args(argv)

    result = convert(pd.read_csv(args.inventory), args.raw, args.bids)
    if args.subjects:
        args.subjects.parent.mkdir(parents=True, exist_ok=True)
        result.drop(columns="dirs").to_csv(args.subjects, index=False)

    counts = result["status"].value_counts().to_dict()
    print(", ".join(f"{n} {status}" for status, n in counts.items()))
    return 0 if counts.get("converted", 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
