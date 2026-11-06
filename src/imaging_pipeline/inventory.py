"""Inventory a folder of DICOM files.

Walks a directory, reads each file's header (no pixel data), and produces one
row per series with the fields needed to plan a conversion: who/what/when,
scanner, sequence parameters and file counts. Files that aren't valid DICOM
are counted separately rather than failing the run.

Usage:
    python -m imaging_pipeline.inventory data/raw --out outputs/inventory.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.multival import MultiValue

# Header fields captured per series. Keyword -> output column name.
SERIES_FIELDS: dict[str, str] = {
    "StudyInstanceUID": "study_uid",
    "SeriesInstanceUID": "series_uid",
    "Modality": "modality",
    "Manufacturer": "manufacturer",
    "ManufacturerModelName": "model",
    "MagneticFieldStrength": "field_strength",
    "SeriesNumber": "series_number",
    "SeriesDescription": "series_description",
    "ProtocolName": "protocol_name",
    "SequenceName": "sequence_name",
    "RepetitionTime": "repetition_time_ms",
    "EchoTime": "echo_time_ms",
    "SliceThickness": "slice_thickness_mm",
    "Rows": "rows",
    "Columns": "columns",
}


@dataclass
class InventoryResult:
    """Series table plus the files that couldn't be read as DICOM."""

    series: pd.DataFrame
    skipped: list[Path] = field(default_factory=list)


def _value(ds: pydicom.Dataset, keyword: str):
    """Return a header value as a plain Python type, or None if absent."""
    value = ds.get(keyword)
    if value is None or value == "":
        return None
    if isinstance(value, MultiValue):
        return "\\".join(str(v) for v in value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):  # includes pydicom's IS
        return int(value)
    if isinstance(value, float):  # includes pydicom's DSfloat
        return float(value)
    return str(value)


def inventory(root: str | Path) -> InventoryResult:
    """Scan ``root`` recursively and summarise every DICOM series found."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    series: dict[str, dict] = {}
    file_counts: dict[str, int] = defaultdict(int)
    skipped: list[Path] = []

    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except (InvalidDicomError, OSError):
            skipped.append(path)
            continue

        uid = _value(ds, "SeriesInstanceUID") or f"no-series-uid:{path.parent}"
        file_counts[uid] += 1
        if uid not in series:
            row = {col: _value(ds, kw) for kw, col in SERIES_FIELDS.items()}
            row["series_uid"] = uid
            row["first_file"] = str(path.relative_to(root))
            series[uid] = row

    columns = list(SERIES_FIELDS.values()) + ["n_files", "first_file"]
    table = pd.DataFrame(
        [{**row, "n_files": file_counts[uid]} for uid, row in series.items()],
        columns=columns,
    )
    if not table.empty:
        table = table.sort_values(["study_uid", "series_number"], na_position="last")
        table = table.reset_index(drop=True)
    return InventoryResult(series=table, skipped=skipped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory a folder of DICOM files.")
    parser.add_argument("root", type=Path, help="Folder to scan")
    parser.add_argument("--out", type=Path, help="CSV file to write (default: print)")
    args = parser.parse_args(argv)

    result = inventory(args.root)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.series.to_csv(args.out, index=False)
        print(f"Wrote {len(result.series)} series to {args.out}")
    else:
        print(result.series.to_string(index=False))
    if result.skipped:
        print(f"Skipped {len(result.skipped)} non-DICOM file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
