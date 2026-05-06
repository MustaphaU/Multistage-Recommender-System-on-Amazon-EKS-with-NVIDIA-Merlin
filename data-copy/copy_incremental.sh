#!/usr/bin/env bash

PV_LOC=${1:-"/var/lib/data"}
AWS_REGION=${2:-"us-east-1"}
S3_BUCKET=${3:-"multistage-feast-bucket"}
NEW_DATA_S3_PATH=${4:-"new_data/"}
REPLAY_DATA_S3_PATH=${5:-"old_merged/"}

set +e
triton_status=$(helm status triton 2>&1)
set -e
if [[ "$triton_status" != *"STATUS: deployed"* ]]; then
    echo "Triton not deployed. Skipping incremental copy and exiting."
    exit 0
fi
echo "Triton is deployed. Proceeding with incremental copy."

echo "downloading new artifacts to $PV_LOC"

# new day's raw interaction data
echo "Downloading new day raw data..."
mkdir -p $PV_LOC/incremental/new_data
rm -rf $PV_LOC/incremental/new_data/*
aws s3 cp --recursive "s3://${S3_BUCKET}/${NEW_DATA_S3_PATH}" "$PV_LOC/incremental/new_data/" --region "$AWS_REGION"

# replay data (sliding window of feature-merged day parquets)
echo "Downloading replay data..."
mkdir -p $PV_LOC/incremental/old_merged
rm -rf $PV_LOC/incremental/old_merged/*
aws s3 cp --recursive "s3://${S3_BUCKET}/${REPLAY_DATA_S3_PATH}" "$PV_LOC/incremental/old_merged/" --region "$AWS_REGION"

echo "copy incremental: done"
echo ""
echo "Everything else is already on the PVC from the initial run:"
echo "  models/two_tower_model/                          <- two-tower checkpoint"
echo "  models/dlrm/                                     <- DLRM checkpoint"
echo "  processed_data/lookup_embeddings/                <- lookup_embeddings_image/text.npy"
echo "  processed_data/processed_nvt/full_workflow/      <- fitted NVT workflow (paths intact)"
echo "  script/feast_repo/                               <- feast repo with credentials injected for feature fetching in fine-tuning"
