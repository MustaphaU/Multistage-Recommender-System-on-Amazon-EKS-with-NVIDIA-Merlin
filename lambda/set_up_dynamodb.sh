ITEMS_TABLE_NAME=${1:-"items"}
REGION=${2:-"us-east-1"}
S3_BUCKET=${3:-"item-images-with-uuid-bucket"}

#create the bucket if it doesn't exist
aws s3 mb s3://$S3_BUCKET --region $REGION || true

aws dynamodb create-table \
    --table-name $ITEMS_TABLE_NAME \
    --attribute-definitions AttributeName=item_id,AttributeType=N \
    --key-schema AttributeName=item_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region $REGION

aws dynamodb wait table-exists --table-name $ITEMS_TABLE_NAME --region $REGION

#bulk-upload all images to S3 in parallel (much faster than per-row uploads)
aws s3 sync raw_and_mappings/item_images_with_UUID/ \
    s3://$S3_BUCKET/items/ \
    --content-type image/jpeg \
    --region $REGION

#run the loader script to populate the DynamoDB table with item metadata
python3 load_items_to_dynamodb.py \
    --items-csv raw_and_mappings/items.csv \
    --mapping-parquet raw_and_mappings/item_id_mapping.parquet \
    --s3-bucket $S3_BUCKET \
    --dynamo-table $ITEMS_TABLE_NAME \
    --aws-region $REGION