#!/usr/bin/env bash
set -euo pipefail

PV_LOC=${1:-"/var/lib/data"}

set +e
triton_status=$(helm status triton 2>&1)
set -e

if [[ "$triton_status" != *"STATUS: deployed"* ]]; then
    echo "Triton not deployed. Skipping incremental preprocessing."
    exit 0
fi

echo "Triton is deployed. Refreshing scripts on PVC and running incremental preprocessing..."
cp -r /script/preprocessing_scripts/. "$PV_LOC/script/preprocessing_scripts/"
cp -r /script/training_scripts/. "$PV_LOC/script/training_scripts/"

python3 -u "$PV_LOC/script/preprocessing_scripts/preprocess_incremental.py" \
    --new_data_path   "$PV_LOC/incremental/new_data" \
    --old_merged_path "$PV_LOC/incremental/old_merged" \
    --workflow_path   "$PV_LOC/processed_data/processed_nvt/full_workflow" \
    --feast_repo_path "$PV_LOC/script/feast_repo/feature_repo" \
    --output_path     "$PV_LOC/incremental/processed_nvt"

echo "Incremental preprocessing complete."
