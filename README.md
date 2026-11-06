# Imaging data pipeline

Takes raw medical imaging data (DICOM) through inventory, conversion to BIDS,
de-identification and quality control to analysis-ready data, with a
dashboard of the results. Public data only.

## Stages

| # | Stage | Tool | Status |
|---|-------|------|--------|
| 1 | Raw DICOM | dcm_qa (Siemens, GE, Philips) | ✅ download script |
| 2 | Inventory | pydicom | ✅ |
| 3 | Convert to BIDS | dcm2niix + HeuDiConv, BIDS Validator | ✅ |
| 4 | De-identify | metadata scrub + defacing (ANTsPy, pydeface mask) | ✅ |
| 5 | Quality control | MRIQC | ⬜ |
| 6 | Preprocess | fMRIPrep | ⬜ |
| 7 | Results store | DuckDB / Parquet | ⬜ |
| 8 | QC dashboard | Dash | ⬜ |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest

./scripts/get_test_data.sh
python -m imaging_pipeline.inventory data/raw --out outputs/inventory.csv
python -m imaging_pipeline.bids outputs/inventory.csv --raw data/raw \
    --bids data/bids --subjects outputs/bids_subjects.csv
python -m imaging_pipeline.validate data/bids --out outputs/bids_validation.json
python -m imaging_pipeline.deidentify data/bids --out data/bids_deid \
    --report outputs/deid_report.csv --qc outputs/deface_qc
python -m imaging_pipeline.validate data/bids_deid --out outputs/deid_validation.json
```

Or run every stage with `snakemake --cores 1`. The validator uses a local
`bids-validator` if installed, otherwise the `bids/validator` Docker image.

## Stage 3: conversion to BIDS

The inventory is the plan: each DICOM study becomes one BIDS subject, and
HeuDiConv runs once per study, so one bad study is recorded as failed without
stopping the rest. The heuristic
([src/imaging_pipeline/heuristic.py](src/imaging_pipeline/heuristic.py))
sorts series into T1w, dwi, bold and epi fieldmaps by series description and
sequence name, and skips everything else.

- Subject labels (`siemens01`, `ge02`, ...) are generated, never taken from
  patient IDs. `outputs/bids_subjects.csv` maps them back to study UIDs, so
  treat it as a re-identification key.
- After conversion, scanner-derived extras (e.g. Philips ADC maps) are removed,
  NIfTI header TRs are set to match the sidecar when a series has a gap between
  volumes, and HeuDiConv's undocumented `scans.tsv` columns are dropped. Each
  change is listed in the `notes` column of `bids_subjects.csv`.
- Functional runs use the placeholder task label `unknown`; the test data has no
  task design.

On the dcm_qa data, 10 of 11 studies convert and the dataset validates with no
errors. The Philips `Rosetta_2013` study fails because its dates are anonymised
to zeros, which HeuDiConv 1.5 can't parse. The remaining validator warnings are
expected for scanner test data (no events files, fieldmaps not linked to runs,
optional sidecar fields).

## Stage 4: de-identification

Writes a de-identified copy (`data/bids_deid`) of the stage 3 dataset, which
stays as it was. Later stages and anything shared should use the copy;
`data/bids` and `outputs/bids_subjects.csv` remain identifiable.

- **Metadata:** HeuDiConv's working folder (patient IDs, dates, UIDs) isn't
  copied. Sidecar keys naming a patient, site, device, operator or date are
  removed, as are free-text comments and study descriptions. Scan dates become
  1925-01-01 (the BIDS convention) with the time of day kept, and NIfTI header
  text is blanked. `SeriesDescription` and `ProtocolName` are kept for QC; on
  real data, check that nobody typed a name into them at the scanner.
- **Faces:** full-head anatomical images (T1w, T2w, FLAIR...) are defaced with
  pydeface's method and face mask, but with ANTsPy doing the alignment instead
  of FSL. FSL is free for non-commercial use only. Every part used here is
  MIT, BSD or Apache licensed, so it can be used commercially. Voxels are
  removed outright, not dimmed, so no faint outline of the face is left.
- **Checks:** the copy is audited for anything that should have been removed,
  and the run fails if anything is found. A before/after picture of each
  defaced image goes to `outputs/deface_qc/`; look at them, because automatic
  defacing can fail on unusual images. The copy is re-validated as BIDS.

The only anatomical image in the dcm_qa data (`sub-ge01` T1w) is a phantom,
not a head, so it can't show whether defacing works. Defacing was checked on a
whole-head MNI template, and the tests check it against a repositioned head
template with a known face mask.

**Skull-stripped images** have no face, and defacing them can cut into the
brain, so each anatomical image is checked first and stripped ones are left
alone. An image counts as stripped only if all three hold: at least 30% of
voxels are exactly zero (raw scans have background noise), the tissue measures
0.3–2.5 L (a brain, not a head), and no more than 1% of the image's edge
contains tissue (a stripped brain sits inside the frame, while a cropped head
runs off it). The check leans towards defacing, because a missed face leaks
identity while a wrongly defaced image only loses data. A `SkullStripped`
true/false value in the image's sidecar JSON overrides the measurements. The
reason for each decision is recorded in `outputs/deid_report.csv`, and stripped
images still get a QC picture so a person can confirm there's no face.

## Data

No client, patient or employer data. Raw data and outputs are git-ignored.
