"""Run the BIDS Validator on a dataset and summarise the result.

Uses a local ``bids-validator`` if one is on PATH, otherwise the
``bids/validator`` Docker image. Writes the validator's full JSON report and
prints a count of issues by severity and code. Exits non-zero if there are
errors; warnings are reported but don't fail the run.

Usage:
    python -m imaging_pipeline.validate data/bids --out outputs/bids_validation.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

DOCKER_IMAGE = "bids/validator:latest"


def validator_command(bids_dir: Path) -> list[str]:
    local = shutil.which("bids-validator")
    if local:
        return [local, str(bids_dir), "--json"]
    if shutil.which("docker"):
        return [
            "docker", "run", "--rm", "--platform", "linux/amd64",
            "-v", f"{bids_dir.resolve()}:/data:ro", DOCKER_IMAGE, "/data", "--json",
        ]  # fmt: skip
    raise FileNotFoundError("Neither bids-validator nor docker is available")


def summarise(report: dict) -> Counter:
    """Count issues by (severity, code)."""
    issues = report.get("issues", {}).get("issues", [])
    return Counter((i.get("severity", "?"), i.get("code", "?")) for i in issues)


def validate(bids_dir: str | Path) -> dict:
    bids_dir = Path(bids_dir)
    if not bids_dir.is_dir():
        raise NotADirectoryError(f"Not a folder: {bids_dir}")
    result = subprocess.run(validator_command(bids_dir), capture_output=True, text=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Validator gave no JSON report:\n{result.stderr[-2000:]}") from err


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a BIDS dataset.")
    parser.add_argument("bids_dir", type=Path)
    parser.add_argument("--out", type=Path, help="Where to write the full JSON report")
    args = parser.parse_args(argv)

    report = validate(args.bids_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))

    counts = summarise(report)
    for (severity, code), n in sorted(counts.items()):
        print(f"{severity:8} {code:40} {n}")
    n_errors = sum(n for (severity, _), n in counts.items() if severity == "error")
    print(f"{n_errors} error(s), {sum(counts.values()) - n_errors} warning(s)")
    return 1 if n_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
