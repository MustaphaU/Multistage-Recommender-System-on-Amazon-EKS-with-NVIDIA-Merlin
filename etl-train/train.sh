PV_LOC=${1:-"/var/lib/data"}
AWS_REGION=${2:-"us-east-1"}
S3_BUCKET=${3:-"multistage-feast-bucket"}
REDIS_HOST=${4:-"multistage-redis.xxxxxx.use1.cache.amazonaws.com"}
REDIS_PORT=${5:-6379}
RETRIEVAL_TOPK=${6:-300}
RANKING_TOPK=${7:-100}
DIVERSITY_MODE=${8:-"false"}

set +e
triton_status=$(helm status triton 2>&1)
set -e

if [[ "$triton_status" != *"STATUS: deployed"* ]]; then
    echo "FIRST RUN: Triton is not running."
    echo "Starting Two-Tower training"
    python3 -u "$PV_LOC/script/training_scripts/train_retrieval_and_setup_faiss.py" \
        --input_path "$PV_LOC/processed_data" \
        --output_path "$PV_LOC/models"

    echo "Training ranking (DLRM) model"
    python3 -u "$PV_LOC/script/training_scripts/train_ranking.py" \
        --input_path "$PV_LOC/processed_data/" \
        --output_path "$PV_LOC/models"

    echo "Assembling the Triton model repository"
    mkdir -p $PV_LOC/triton_model_repository
    cp -r $PV_LOC/script/triton_repository_template/* $PV_LOC/triton_model_repository
    cp -r $PV_LOC/processed_data/processed_nvt/user_subworkflow/* $PV_LOC/triton_model_repository/nvt_user_transform/1/workflow
    cp -r $PV_LOC/processed_data/processed_nvt/item_subworkflow/* $PV_LOC/triton_model_repository/nvt_item_transform/1/workflow
    cp -r $PV_LOC/processed_data/processed_nvt/context_subworkflow/* $PV_LOC/triton_model_repository/nvt_context_transform/1/workflow
    cp -r $PV_LOC/models/query_tower/* $PV_LOC/triton_model_repository/query_tower/1
    cp -r $PV_LOC/models/faiss_index/index.faiss $PV_LOC/triton_model_repository/faiss_retrieval/1/index.faiss
    cp -r $PV_LOC/processed_data/lookup_embeddings/*.npy $PV_LOC/triton_model_repository/multimodal_embedding_lookup/1
    cp -r $PV_LOC/models/dlrm/* $PV_LOC/triton_model_repository/dlrm_ranking/1
    cp $PV_LOC/processed_data/processed_nvt/item_subworkflow/categories/unique.item_id.parquet $PV_LOC/triton_model_repository/item_id_decoder/1/categories/
    mkdir -p $PV_LOC/triton_model_repository/feast_item_lookup/1/categories
    cp $PV_LOC/processed_data/processed_nvt/item_subworkflow/categories/unique.item_id.parquet $PV_LOC/triton_model_repository/feast_item_lookup/1/categories/
    
    AGE_MEAN=$(python3 -c "import json; print(int(json.load(open('$PV_LOC/processed_data/processed_nvt/age_stats.json')).get('age_mean', 36)))")

	cat >> "$PV_LOC/triton_model_repository/feast_user_lookup/config.pbtxt" <<-EOF
	parameters {
	    key: "DEFAULT_USER_AGE"
	    value { string_value: "$AGE_MEAN" }
	}
	EOF

	REDIS_URL="rediss://${REDIS_HOST}:${REDIS_PORT}/1"
	cat >> "$PV_LOC/triton_model_repository/filter_seen_items/config.pbtxt" <<-EOF
	parameters {
	    key: "REDIS_URL"
	    value { string_value: "$REDIS_URL" }
	}
	EOF

	cat >> "$PV_LOC/triton_model_repository/faiss_retrieval/config.pbtxt" <<-EOF
	parameters {
	    key: "RETRIEVAL_TOPK"
	    value { string_value: "$RETRIEVAL_TOPK" }
	}
	EOF

	cat >> "$PV_LOC/triton_model_repository/softmax_sampling/config.pbtxt" <<-EOF
	parameters {
	    key: "RANKING_TOPK"
	    value { string_value: "$RANKING_TOPK" }
	}
	parameters {
	    key: "DIVERSITY_MODE"
	    value { string_value: "$DIVERSITY_MODE" }
	}
	EOF

    echo "Uploading directories to S3"
    aws s3 cp --recursive $PV_LOC/triton_model_repository s3://${S3_BUCKET}/triton_model_repository --region $AWS_REGION
    aws s3 cp --recursive $PV_LOC/models/two_tower_model s3://${S3_BUCKET}/two_tower_model --region $AWS_REGION
    aws s3 cp --recursive $PV_LOC/processed_data/processed_nvt/full_workflow s3://${S3_BUCKET}/full_workflow --region $AWS_REGION
    aws s3 cp --recursive $PV_LOC/raw_data/masked_train s3://${S3_BUCKET}/old_merged --region $AWS_REGION
    aws s3 cp --recursive $PV_LOC/raw_data/ s3://${S3_BUCKET}/old_merged --exclude "*" --include "valid_day_*.parquet" --region $AWS_REGION
    aws s3 cp --recursive $PV_LOC/script/feast_repo s3://${S3_BUCKET}/feast_repo --region $AWS_REGION
    aws s3 cp --recursive $PV_LOC/script/feast_repo s3://${S3_BUCKET}/feast_repo --region $AWS_REGION
else
    echo "Triton is already running. This is a full retrain run."
    #TODO: add the logic here
fi
