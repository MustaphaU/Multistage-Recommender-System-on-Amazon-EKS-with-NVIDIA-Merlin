"""
load_items_to_dynamodb.py — Load item catalog into DynamoDB for the demo.

Steps:
  1. Read items.csv (raw items keyed by UUID).
  2. Read item_id_mapping.parquet (UUID -> sklearn label-encoded integer).
  3. Write each item to DynamoDB with the encoded integer as partition key.

Note:
  Images should be bulk-uploaded to S3 separately (much faster) using:
    aws s3 sync <images-dir>/ s3://<bucket>/<prefix>/ --content-type image/jpeg

Prereqs:
  - DynamoDB table created with partition key 'item_id' (Number).
  - S3 bucket exists with images already uploaded.
  - AWS credentials configured (env, ~/.aws/credentials, or IAM role).

Usage:
  python3 load_items_to_dynamodb.py \
      --items-csv raw_and_mappings/items.csv \
      --mapping-parquet raw_and_mappings/item_id_mapping.parquet \
      --s3-bucket my-recsys-demo-images \
      --s3-prefix items \
      --dynamo-table items \
      --aws-region us-east-1
"""

import argparse
from decimal import Decimal

import boto3
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-csv",       required=True)
    parser.add_argument("--mapping-parquet", required=True)
    parser.add_argument("--s3-bucket",       required=True)
    parser.add_argument("--s3-prefix",       default="items")
    parser.add_argument("--dynamo-table",    required=True)
    parser.add_argument("--aws-region",      default="us-east-1")
    args = parser.parse_args()

    items   = pd.read_csv(args.items_csv)
    mapping = pd.read_parquet(args.mapping_parquet)

    # Join: raw ITEM_ID (UUID) -> encoded integer
    df = items.merge(
        mapping.rename(columns={"item_id": "ITEM_ID", "encoded": "encoded_id"}),
        on="ITEM_ID", how="inner",
    )
    print(f"Joined rows: {len(df)} (items: {len(items)}, mapping: {len(mapping)})")

    dynamodb = boto3.resource("dynamodb", region_name=args.aws_region)
    table    = dynamodb.Table(args.dynamo_table)

    written = 0
    with table.batch_writer() as batch:
        for _, row in df.iterrows():
            uuid      = row["ITEM_ID"]
            s3_key    = f"{args.s3_prefix}/{uuid}.jpg"
            image_url = f"https://{args.s3_bucket}.s3.amazonaws.com/{s3_key}"

            batch.put_item(Item={
                "item_id":     int(row["encoded_id"]),
                "item_uuid":   uuid,
                "title":       str(row["PRODUCT_NAME"]),
                "description": str(row["PRODUCT_DESCRIPTION"]),
                "category_l1": str(row["CATEGORY_L1"]),
                "category_l2": str(row["CATEGORY_L2"]),
                "gender":      str(row["GENDER"]),
                "promoted":    str(row["PROMOTED"]),
                "price":       Decimal(str(row["PRICE"])),
                "image_url":   image_url,
            })
            written += 1
            if written % 500 == 0:
                print(f"  wrote {written} items...")

    print(f"Done. Written: {written}")


if __name__ == "__main__":
    main()