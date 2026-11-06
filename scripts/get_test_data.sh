#!/usr/bin/env bash
# Download small public multi-vendor DICOM test sets (dcm_qa) into data/raw.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
for repo in dcm_qa dcm_qa_philips dcm_qa_ge; do
  if [ ! -d "data/raw/$repo" ]; then
    git clone --depth 1 "https://github.com/neurolabusc/$repo.git" "data/raw/$repo"
  fi
done
echo "Test data is in data/raw"
