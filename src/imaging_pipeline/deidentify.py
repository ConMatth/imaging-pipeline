"""De-identify a BIDS dataset: scrub metadata and deface anatomical images.

Writes a de-identified copy of the stage 3 dataset; the source is not changed.

What's removed or changed:
- HeuDiConv's working folder (``.heudiconv/``), which holds patient IDs, ages,
  dates, accession numbers and DICOM UIDs. It is not copied.
- Sidecar JSON keys that name or locate a person, site or device (patient,
  institution, physician, operator, station, serial number, dates, UIDs, free-text
  comments, study and procedure descriptions).
- Dates in ``*_scans.tsv``: set to 1925-01-01, the BIDS convention for removed
  dates. Time of day is kept so the scan order stays visible.
- NIfTI header free-text fields (``descrip``, ``aux_file``).
- Faces, in full-head anatomical images (T1w, T2w, FLAIR...). This follows
  pydeface's method, using pydeface's template and face mask (MIT), but with
  ANTsPy (Apache-2.0) for the alignment instead of FSL, whose licence doesn't
  allow commercial use for free. A before/after picture of each defaced image
  is saved so a person can check the face is gone. Images that are already
  skull-stripped have no face and are left alone (see ``check_skull_stripped``).

After writing, the copy is audited for anything this module should have
removed. The run fails if the audit finds anything.

SeriesDescription and ProtocolName are kept because QC needs them. They're
typed in at the scanner, so check them by hand on real data.

Usage:
    python -m imaging_pipeline.deidentify data/bids --out data/bids_deid \\
        --report outputs/deid_report.csv --qc outputs/deface_qc
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import stat
from importlib.resources import files
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd

# Sidecar keys containing any of these (case-insensitive) are removed.
IDENTIFYING_KEY_PARTS = (
    "patient", "institution", "physician", "operator", "station", "serial",
    "date", "uid", "accession", "address", "birth", "comment",
    "studydescription", "procedurestepdescription",
)  # fmt: skip
# ...except these, which describe the scan, not the person.
KEEP_KEYS = {"PatientPosition", "ImageOrientationPatientDICOM"}

REMOVED_DATE = "1925-01-01"
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")

# Full-head anatomical images: the ones with a face in them.
_FACE_IMAGE = re.compile(r"_(T1w|T2w|PDw|T2starw|FLAIR|UNIT1|inplaneT1|inplaneT2)\.nii(\.gz)?$")

_NOT_COPIED = shutil.ignore_patterns(".heudiconv", ".bidsignore", ".gitkeep")


def is_identifying_key(key: str) -> bool:
    lowered = key.lower()
    return key not in KEEP_KEYS and any(part in lowered for part in IDENTIFYING_KEY_PARTS)


def scrub_sidecar(path: Path) -> list[str]:
    """Remove identifying keys from a sidecar JSON. Returns the keys removed."""
    data = json.loads(path.read_text())
    removed = sorted(k for k in data if is_identifying_key(k))
    if removed:
        kept = {k: v for k, v in data.items() if k not in removed}
        path.write_text(json.dumps(kept, indent=2) + "\n")
    return removed


def scrub_scans_tsv(path: Path) -> int:
    """Replace the date in acq_time with the BIDS removed-date value. Returns rows changed."""
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "acq_time" not in table:
        return 0
    has_date = table["acq_time"].str.match(_DATE)
    table.loc[has_date, "acq_time"] = REMOVED_DATE + table.loc[has_date, "acq_time"].str[10:]
    table.to_csv(path, sep="\t", index=False)
    return int(has_date.sum())


def _save_like(image: nb.Nifti1Image, data: np.ndarray, header, path: Path) -> None:
    out = nb.Nifti1Image(data, image.affine, header)
    out.set_data_dtype(image.get_data_dtype())
    nb.save(out, path)


def scrub_nifti_header(path: Path) -> bool:
    """Blank the header's free-text fields. Returns True if anything changed."""
    image = nb.load(path)
    header = image.header.copy()
    if not (header["descrip"].item() or header["aux_file"].item()):
        return False
    header["descrip"] = b""
    header["aux_file"] = b""
    _save_like(image, np.asanyarray(image.dataobj), header, path)
    return True


# Skull-stripped images are just the brain, surrounded by exact zeros. All three
# must hold before defacing is skipped, because missing a face leaks identity
# while defacing a stripped image only damages data (and the QC picture shows it).
MIN_ZERO_FRACTION = 0.3  # raw scans have a background of noise, so few exact zeros
BRAIN_LITRES = (0.3, 2.5)  # infant to large adult brain; whole heads are larger
MAX_EDGE_FRACTION = 0.01  # a stripped brain doesn't reach the image's edges; a head does


def check_skull_stripped(path: Path) -> tuple[bool, str]:
    """Decide whether an anatomical image is already skull-stripped. Returns (answer, reason).

    A ``SkullStripped`` value in the sidecar JSON is used as given, which also
    makes it a manual override. Otherwise the image itself is measured.
    """
    sidecar = Path(str(path).replace(".nii.gz", ".json").replace(".nii", ".json"))
    if sidecar.exists():
        value = json.loads(sidecar.read_text()).get("SkullStripped")
        if isinstance(value, bool):
            return value, f"sidecar says SkullStripped={str(value).lower()}"

    image = nb.load(path)
    data = np.asanyarray(image.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    voxel_litres = float(np.prod(image.header.get_zooms()[:3])) / 1e6
    zero_fraction = float(np.mean(data == 0))
    tissue_litres = np.count_nonzero(data) * voxel_litres
    faces = [data[0], data[-1], data[:, 0], data[:, -1], data[:, :, 0], data[:, :, -1]]
    edge_fraction = sum(np.count_nonzero(f) for f in faces) / sum(f.size for f in faces)

    stripped = (
        zero_fraction >= MIN_ZERO_FRACTION
        and BRAIN_LITRES[0] <= tissue_litres <= BRAIN_LITRES[1]
        and edge_fraction <= MAX_EDGE_FRACTION
    )
    reason = (
        f"{zero_fraction:.0%} zeros, {tissue_litres:.2f} L of tissue, "
        f"{edge_fraction:.1%} tissue at image edges"
    )
    return stripped, reason


def face_mask(path: Path) -> np.ndarray:
    """Weights in the image's voxel grid: 1 keeps a voxel, 0 is face (or outside the template).

    The pydeface template is aligned to the image with an affine registration,
    and its face mask is carried across with the same transform.
    """
    import ants  # imported here: slow to load, and only needed for defacing

    data = files("pydeface") / "data"
    image = ants.image_read(str(path))
    template = ants.image_read(str(data / "mean_reg2mean.nii.gz"))
    mask = ants.image_read(str(data / "facemask.nii.gz"))
    registration = ants.registration(
        fixed=image,
        moving=template,
        type_of_transform="Affine",
        aff_metric="mattes",
        random_seed=0,
    )
    warped = ants.apply_transforms(
        fixed=image,
        moving=mask,
        transformlist=registration["fwdtransforms"],
        interpolator="linear",
        defaultvalue=0,
    )
    return warped.numpy()


def deface(path: Path, qc_png: Path | None = None) -> float:
    """Deface an image in place. Returns the fraction of non-zero voxels removed."""
    image = nb.load(path)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image to deface, got {image.ndim}D: {path}")
    original = image.get_fdata()
    mask = face_mask(path)
    if mask.shape != original.shape:
        raise RuntimeError(f"Face mask shape {mask.shape} doesn't match image {original.shape}")
    # Keep or remove each voxel outright. pydeface multiplies by the interpolated
    # mask, which only dims voxels at the face's edge and can leave its outline.
    defaced = np.where(mask >= 0.5, original, 0)
    _save_like(image, defaced, image.header.copy(), path)

    if qc_png is not None:
        save_deface_picture(image, original, defaced, qc_png)
    nonzero = np.count_nonzero(original)
    return float(1 - np.count_nonzero(defaced) / nonzero) if nonzero else 0.0


def save_deface_picture(image, original: np.ndarray, defaced: np.ndarray | None, png: Path) -> None:
    """Middle sagittal and axial slices, before and after, for a person to check.

    With ``defaced=None`` (an image left alone as skull-stripped), only the
    original is shown, so a person can confirm there's no face.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def canonical(data):
        return nb.as_closest_canonical(nb.Nifti1Image(data, image.affine)).get_fdata()

    rows = [("original", canonical(original))]
    if defaced is not None:
        rows.append(("defaced", canonical(defaced)))
    else:
        rows[0] = ("not defaced (skull-stripped)", rows[0][1])
    x, _, z = (s // 2 for s in rows[0][1].shape)
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 4 * len(rows)), squeeze=False)
    for row, (label, volume) in enumerate(rows):
        axes[row, 0].imshow(np.rot90(volume[x, :, :]), cmap="gray")
        axes[row, 1].imshow(np.rot90(volume[:, :, z]), cmap="gray")
        axes[row, 0].set_title(f"{label}: sagittal")
        axes[row, 1].set_title(f"{label}: axial")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(png.stem)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=80)
    plt.close(fig)


def find_identifiers(dataset: Path) -> list[str]:
    """Anything left in a de-identified dataset that should have been removed."""
    problems = []
    for path in sorted(dataset.rglob("*")):
        rel = path.relative_to(dataset)
        if ".heudiconv" in rel.parts:
            problems.append(f"{rel}: HeuDiConv working files")
        elif path.name.endswith(".json") and path.is_file():
            keys = [k for k in json.loads(path.read_text()) if is_identifying_key(k)]
            if keys:
                problems.append(f"{rel}: identifying keys {keys}")
        elif path.name.endswith("_scans.tsv"):
            table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
            times = table.get("acq_time", pd.Series(dtype=str))
            if (times.str.match(_DATE) & ~times.str.startswith(REMOVED_DATE)).any():
                problems.append(f"{rel}: real scan dates")
        elif path.name.endswith((".nii", ".nii.gz")):
            header = nb.load(path).header
            if header["descrip"].item() or header["aux_file"].item():
                problems.append(f"{rel}: text in NIfTI header")
    return problems


def _prepare_output(source: Path, out: Path) -> None:
    """Remove a previous run's output. Refuses to delete anything that isn't one."""
    if out.resolve() == source.resolve() or source.resolve() in out.resolve().parents:
        raise ValueError(f"Output {out} must be outside the source dataset {source}")
    if out.exists() and any(out.iterdir()):
        if not (out / "dataset_description.json").exists():
            raise FileExistsError(f"{out} exists and isn't a dataset this tool wrote; not deleting")
        shutil.rmtree(out)


def deidentify(
    source: str | Path, out: str | Path, qc_dir: str | Path | None = None
) -> pd.DataFrame:
    """Write a de-identified copy of ``source`` to ``out``. Returns one row per change."""
    source, out = Path(source), Path(out)
    if not (source / "dataset_description.json").exists():
        raise FileNotFoundError(f"{source} is not a BIDS dataset (no dataset_description.json)")
    _prepare_output(source, out)
    shutil.copytree(source, out, ignore=_NOT_COPIED, dirs_exist_ok=True)
    for path in out.rglob("*"):  # HeuDiConv outputs are read-only; the copy needs editing
        path.chmod(path.stat().st_mode | stat.S_IWUSR)

    changes = []
    if (source / ".heudiconv").exists():
        changes.append(("", "removed .heudiconv/ working files"))
    for path in sorted(out.rglob("*.json")):
        removed = scrub_sidecar(path)
        if removed:
            changes.append((path, f"removed keys: {', '.join(removed)}"))
    for path in sorted(out.rglob("*_scans.tsv")):
        n = scrub_scans_tsv(path)
        if n:
            changes.append((path, f"replaced dates in {n} row(s)"))
    for path in sorted(out.rglob("*.nii*")):
        if scrub_nifti_header(path):
            changes.append((path, "blanked NIfTI header text"))
        if path.parent.name == "anat" and _FACE_IMAGE.search(path.name):
            qc_png = Path(qc_dir) / f"{path.name.split('.')[0]}.png" if qc_dir else None
            stripped, reason = check_skull_stripped(path)
            if stripped:
                if qc_png is not None:
                    image = nb.load(path)
                    save_deface_picture(image, image.get_fdata(), None, qc_png)
                changes.append((path, f"not defaced: already skull-stripped ({reason})"))
            else:
                removed = deface(path, qc_png)
                changes.append(
                    (path, f"defaced ({removed:.1%} of non-zero voxels removed; {reason})")
                )

    description_path = out / "dataset_description.json"
    description = json.loads(description_path.read_text())
    description.setdefault("GeneratedBy", []).append(
        {"Name": "imaging-pipeline deidentify", "Description": "metadata scrub + ANTsPy defacing"}
    )
    description_path.write_text(json.dumps(description, indent=2) + "\n")

    return pd.DataFrame(
        [(str(p.relative_to(out)) if p else "", c) for p, c in changes], columns=["file", "change"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="De-identify a BIDS dataset.")
    parser.add_argument("source", type=Path, help="BIDS dataset from stage 3")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the copy")
    parser.add_argument("--report", type=Path, help="CSV listing every change")
    parser.add_argument("--qc", type=Path, help="Folder for before/after defacing pictures")
    args = parser.parse_args(argv)

    changes = deidentify(args.source, args.out, args.qc)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        changes.to_csv(args.report, index=False)
    face_rows = changes["change"].str.contains("defaced")
    for row in changes[face_rows].itertuples():
        print(f"{row.file}: {row.change}")
    print(f"{len(changes)} change(s) written to {args.out}")

    problems = find_identifiers(args.out)
    for problem in problems:
        print(f"STILL IDENTIFYING: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
