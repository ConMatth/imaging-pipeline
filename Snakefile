# Pipeline orchestration. Each stage is added as a rule as it's built.
configfile: "config/config.yaml"

RAW = config["raw_dir"]
BIDS = config["bids_dir"]
DEID = config["deid_dir"]
OUT = config["outputs_dir"]


rule all:
    input:
        f"{OUT}/bids_validation.json",
        f"{OUT}/deid_validation.json",


# Stage 2: inventory of every DICOM series under data/raw
rule inventory:
    input:
        RAW,
    output:
        f"{OUT}/inventory.csv",
    shell:
        "python -m imaging_pipeline.inventory {input} --out {output}"


# Stage 3: one BIDS subject per DICOM study, via HeuDiConv + dcm2niix.
# bids_subjects.csv links subject labels to study UIDs: keep it private.
rule bids:
    input:
        f"{OUT}/inventory.csv",
    output:
        f"{OUT}/bids_subjects.csv",
    shell:
        "python -m imaging_pipeline.bids {input} --raw {RAW} --bids {BIDS} --subjects {output}"


# Stage 3: BIDS Validator (local install, else Docker). Fails on errors, not warnings.
rule validate_bids:
    input:
        f"{OUT}/bids_subjects.csv",
    output:
        f"{OUT}/bids_validation.json",
    shell:
        "python -m imaging_pipeline.validate {BIDS} --out {output}"


# Stage 4: de-identified copy of the BIDS dataset (metadata scrub + defacing).
# Fails if the audit finds anything identifying left. Check outputs/deface_qc by eye.
rule deidentify:
    input:
        f"{OUT}/bids_validation.json",
    output:
        f"{OUT}/deid_report.csv",
    shell:
        "python -m imaging_pipeline.deidentify {BIDS} --out {DEID} "
        "--report {output} --qc {OUT}/deface_qc"


# Stage 4: the de-identified copy must still be valid BIDS.
rule validate_deid:
    input:
        f"{OUT}/deid_report.csv",
    output:
        f"{OUT}/deid_validation.json",
    shell:
        "python -m imaging_pipeline.validate {DEID} --out {output}"
