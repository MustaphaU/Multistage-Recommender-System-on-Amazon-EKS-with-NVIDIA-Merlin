#!/usr/bin/env bash
set -euo pipefail

PV_LOC=${1:-"/var/lib/data"}
AWS_REGION=${2:-"us-east-1"}
S3_BUCKET=${3:-"multistage-feast-bucket"}
EPOCHS=${4:-1}
LR=${5:-"1e-4"}

set +e
triton_status=$(helm status triton 2>&1)
set -e

if [[ "$triton_status" != *"STATUS: deployed"* ]]; then
    echo "Triton not deployed. Skipping incremental training."
    exit 0
fi

echo "Fine-tuning retrieval (query tower)"
python3 -u "$PV_LOC/script/training_scripts/finetune_retrieval.py" \
    --input_path      "$PV_LOC/incremental/processed_nvt" \
    --embeddings_path "$PV_LOC/processed_data/lookup_embeddings" \
    --checkpoint_path "$PV_LOC/models" \
    --output_path     "$PV_LOC/models" \
    --epochs          "$EPOCHS" \
    --lr              "$LR"

echo "Fine-tuning ranking (DLRM)"
python3 -u "$PV_LOC/script/training_scripts/finetune_ranking.py" \
    --input_path      "$PV_LOC/incremental/processed_nvt" \
    --embeddings_path "$PV_LOC/processed_data/lookup_embeddings" \
    --checkpoint_path "$PV_LOC/models" \
    --output_path     "$PV_LOC/models" \
    --epochs          "$EPOCHS" \
    --lr              "$LR"

echo "Updating Triton model repository"
latest_dlrm=$(ls -v "$PV_LOC/triton_model_repository/dlrm_ranking/" | grep -E '^[0-9]+$' | tail -n1)
new_dlrm=$(($latest_dlrm + 1))
mkdir -p "$PV_LOC/triton_model_repository/dlrm_ranking/$new_dlrm"
cp -r "$PV_LOC/models/dlrm/model.savedmodel" "$PV_LOC/triton_model_repository/dlrm_ranking/$new_dlrm/"
echo "DLRM: v$latest_dlrm -> v$new_dlrm"

latest_qt=$(ls -v "$PV_LOC/triton_model_repository/query_tower/" | grep -E '^[0-9]+$' | tail -n1)
new_qt=$(($latest_qt + 1))
mkdir -p "$PV_LOC/triton_model_repository/query_tower/$new_qt"
cp -r "$PV_LOC/models/query_tower/model.savedmodel" "$PV_LOC/triton_model_repository/query_tower/$new_qt/"
echo "Query tower: v$latest_qt -> v$new_qt"

echo "Uploading updated models to S3"
aws s3 cp --recursive "$PV_LOC/triton_model_repository/dlrm_ranking/$new_dlrm" \
    "s3://${S3_BUCKET}/triton_model_repository/dlrm_ranking/$new_dlrm/" --region "$AWS_REGION"
aws s3 cp --recursive "$PV_LOC/triton_model_repository/query_tower/$new_qt" \
    "s3://${S3_BUCKET}/triton_model_repository/query_tower/$new_qt/" --region "$AWS_REGION"

#we overwrite the two_tower_model since it is only used for initializing training.
aws s3 cp --recursive "$PV_LOC/models/two_tower_model" \
    "s3://${S3_BUCKET}/two_tower_model/" --region "$AWS_REGION"

echo "Updating old_merged in S3 for next incremental run"

#rename valid_day_ -> train_day_: they were replayed as training data this run
aws s3 ls "s3://${S3_BUCKET}/old_merged/" | awk '{print $4}' | grep '^valid_day_' | while read obj; do
    new_name="${obj/valid_day_/train_day_}"
    aws s3 mv "s3://${S3_BUCKET}/old_merged/${obj}" "s3://${S3_BUCKET}/old_merged/${new_name}" --region "$AWS_REGION"
    echo "Renamed $obj -> $new_name"
done

DAY_DATE=$(date +%Y%m%d)
aws s3 cp "$PV_LOC/incremental/new_data/train_day_${DAY_DATE}.parquet" \
    "s3://${S3_BUCKET}/old_merged/train_day_${DAY_DATE}.parquet" --region "$AWS_REGION"
aws s3 cp "$PV_LOC/incremental/new_data/valid_day_${DAY_DATE}.parquet" \
    "s3://${S3_BUCKET}/old_merged/valid_day_${DAY_DATE}.parquet" --region "$AWS_REGION"

echo "Incremental training complete"
