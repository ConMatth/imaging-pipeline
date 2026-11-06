"""HeuDiConv heuristic: decide where each DICOM series goes in the BIDS dataset.

HeuDiConv loads this file by path and calls ``infotodict``. The decision itself
lives in ``classify`` so it can be tested without running HeuDiConv.

Matching uses the series description and sequence name. Protocol name isn't
used because GE puts the same generic protocol ("Brain Advanced Sequences") on
every series, and volume counts aren't used because HeuDiConv reports GE
series as one volume however many there are.

The test data has no task design, so every functional run gets the placeholder
task label "unknown". Change ``FUNC`` when converting a real study.
"""

from __future__ import annotations

import re

ANAT_T1W = "sub-{subject}/anat/sub-{subject}_run-{item:02d}_T1w"
DWI = "sub-{subject}/dwi/sub-{subject}_run-{item:02d}_dwi"
FUNC = "sub-{subject}/func/sub-{subject}_task-unknown_run-{item:02d}_bold"
FMAP_EPI = "sub-{{subject}}/fmap/sub-{{subject}}_dir-{dir}_run-{{item:02d}}_epi"

# Phase-encoding direction written at the end of a series description, e.g. "..._BW2222_AP".
_DIRECTION = re.compile(r"[_ ](AP|PA|LR|RL)$", re.IGNORECASE)

_DWI_WORDS = ("dwi", "dti", "tensor", "dt_")
_T1W_WORDS = ("mprage", "mp-rage", "mp_rage", "t1w", "t1 ")
_BOLD_WORDS = ("fmri", "bold", "epi")
# Sequence names of 2D gradient-echo EPI: Siemens "epfid", GE "epiRT".
_BOLD_SEQUENCES = ("epfid", "epirt")


def classify(seqinfo) -> str | None:
    """Return the BIDS path template for a series, or None to skip it."""
    description = (seqinfo.series_description or "").strip()
    desc = description.lower()
    sequence = (seqinfo.sequence_name or "").lower()

    if seqinfo.is_derived or seqinfo.dim1 == 0:  # derived images, reports, screenshots
        return None
    if any(word in desc for word in _DWI_WORDS):
        return DWI
    if any(word in f"{desc} " for word in _T1W_WORDS):
        return ANAT_T1W
    direction = _DIRECTION.search(description)
    if direction and "epse" in sequence:  # single-volume spin-echo EPI with a stated direction
        return FMAP_EPI.format(dir=direction.group(1).upper())
    if any(word in desc for word in _BOLD_WORDS) or any(s in sequence for s in _BOLD_SEQUENCES):
        return FUNC
    return None


def create_key(template: str, outtype=("nii.gz",), annotation_classes=None):
    if not template:
        raise ValueError("Template must be a valid format string")
    return template, outtype, annotation_classes


def infotodict(seqinfo):
    """HeuDiConv entry point: map each output key to the series that belong to it."""
    info: dict[tuple, list[str]] = {}
    for s in seqinfo:
        template = classify(s)
        if template is not None:
            info.setdefault(create_key(template), []).append(s.series_id)
    return info
