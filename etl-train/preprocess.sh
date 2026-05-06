PV_LOC=${1:-"/var/lib/data"}
AWS_REGION=${2:-"us-east-1"}
S3_BUCKET=${3:-"multistage-feast-bucket"}
GLUE_DATABASE=${4:-"feast_glue_db"}
ATHENA_WORKGROUP=${5:-"feast_materialize_wg"}
S3_INCREMENTAL_PATH=${6:-"path-in-s3-bucket-for-incremental-data"}
REDIS_HOST=${7:-"multistage-redis.xxxxxx.use1.cache.amazonaws.com"}
REDIS_PORT=${8:-6379}


cp -r /script $PV_LOC/
cd $PV_LOC
echo "Working directory: $PV_LOC"

set +e
triton_status=$(helm status triton 2>&1)
set -e

if [[ "$triton_status" != *"STATUS: deployed"* ]]; then
    echo "FIRST RUN: Triton is not running. This is the first run - running the full preprocessing and training"
    echo "preprocessing ..."
    RAW_DATA_PATH="$PV_LOC/raw_data"
    echo "preprocessing data from: $RAW_DATA_PATH the contents are:"
    ls -al $RAW_DATA_PATH

    mkdir -p $PV_LOC/processed_data
    
    python3 -u $PV_LOC/script/preprocessing_scripts/preprocess.py \
        --input_path $RAW_DATA_PATH \
        --base_dir $PV_LOC/processed_data \
        --train_days 9 \
        --valid_days 3

    echo "Setting up the feature store with feast"
    export S3_BUCKET AWS_REGION REDIS_HOST REDIS_PORT GLUE_DATABASE ATHENA_WORKGROUP
    envsubst < "$PV_LOC/script/feast_repo/feature_repo/feature_store.yaml.template" > "$PV_LOC/script/feast_repo/feature_repo/feature_store.yaml"
    envsubst < "$PV_LOC/script/feast_repo/feature_repo/item_features.py.template" > "$PV_LOC/script/feast_repo/feature_repo/item_features.py"
    envsubst < "$PV_LOC/script/feast_repo/feature_repo/user_features.py.template" > "$PV_LOC/script/feast_repo/feature_repo/user_features.py"
    rm -f $PV_LOC/script/feast_repo/feature_repo/feature_store.yaml.template
    rm -f $PV_LOC/script/feast_repo/feature_repo/item_features.py.template
    rm -f $PV_LOC/script/feast_repo/feature_repo/user_features.py.template

    echo "generated the feature_store.yaml:"
    cat $PV_LOC/script/feast_repo/feature_repo/feature_store.yaml

    echo "item_features.py:"
    cat $PV_LOC/script/feast_repo/feature_repo/item_features.py

    echo "user_features.py:"
    cat $PV_LOC/script/feast_repo/feature_repo/user_features.py

    echo "uploading feature parquets to S3.."
    aws s3 cp "$PV_LOC/processed_data/for_feature_store/user_features.parquet" "s3://${S3_BUCKET}/feast/data/user_features/user_features.parquet" --region $AWS_REGION
    aws s3 cp "$PV_LOC/processed_data/for_feature_store/item_features.parquet" "s3://${S3_BUCKET}/feast/data/item_features/item_features.parquet" --region $AWS_REGION

    cd $PV_LOC/script/feast_repo/feature_repo
    feast apply
    timeout 360 feast materialize \
        "1990-01-01T00:00:00" \
        "$(date -u +%Y-%m-%dT%H:%M:%S)"

    echo "feast setup complete"

    cd $PV_LOC
    echo "write popular items to valkey db=3 in ElastiCache for popular/trending items retrieval"
    python3 -u $PV_LOC/script/preprocessing_scripts/dBwrite_popular_items.py \
        --input_path $RAW_DATA_PATH \
        --redis_host $REDIS_HOST \
        --redis_port $REDIS_PORT

else
    echo "Triton is already running. This is a full retrain run"
    #TODO: preprocessing logic
fi



    


    
