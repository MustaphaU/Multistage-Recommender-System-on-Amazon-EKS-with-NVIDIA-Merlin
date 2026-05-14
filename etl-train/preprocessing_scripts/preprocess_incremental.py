import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import glob
import io
import random
import numpy as np
import pandas as pd
import boto3
import feast
import nvtabular as nvt
from nvtabular import Workflow
from merlin.io.dataset import Dataset
import logging
import argparse
logging.disable(logging.WARNING)


def _read_behavioral_updates(bucket: str, region: str, days: int = 7) -> pd.DataFrame:
    """
    Read real-time top_category updates written by the feature_computation Lambda
    from the S3 offline store. Returns the most recent update per user_id.
    """
    s3 = boto3.client("s3", region_name=region)
    paginator = s3.get_paginator("list_objects_v2")
    dfs = []
    for i in range(days):
        date_str = (pd.Timestamp.now() - pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        prefix = f"feast/behavioral_updates/dt={date_str}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                dfs.append(pd.read_parquet(io.BytesIO(body), columns=["user_id", "top_category", "datetime"]))
    if not dfs:
        return pd.DataFrame(columns=["user_id", "top_category_realtime"])
    return (
        pd.concat(dfs, ignore_index=True)
        .sort_values("datetime", ascending=False)
        .drop_duplicates("user_id")[["user_id", "top_category"]]
        .rename(columns={"top_category": "top_category_realtime"})
    )


def run_incremental_preprocessing(new_data_path, old_merged_path, workflow_path,
                                   feast_repo_path, output_path,
                                   feast_s3_bucket=None, feast_aws_region=None):

    new_files = glob.glob(os.path.join(new_data_path, "day_*.parquet"))
    interactions_df = pd.concat([pd.read_parquet(f) for f in sorted(new_files)], ignore_index=True)
    logging.info("Loaded %d new interactions from %d file(s)", len(interactions_df), len(new_files))

    store = feast.FeatureStore(repo_path=feast_repo_path)

    user_entity_df = pd.DataFrame({
        "user_id": interactions_df["user_id"].unique(),
        "event_timestamp": pd.Timestamp.now(tz="UTC"),
    })
    user_features_df = store.get_historical_features(
        entity_df=user_entity_df,
        features=["user_features:age", "user_features:gender", "user_features:top_category"],
    ).to_df().drop(columns=["event_timestamp"])

    # Override Feast historical top_category with real-time behavioral updates from S3.
    # This closes the training/serving skew loop: the Lambda writes to both the online
    # store and S3, so incremental training sees the same top_category as serving.
    if feast_s3_bucket and feast_aws_region:
        behavioral_updates = _read_behavioral_updates(feast_s3_bucket, feast_aws_region)
        if not behavioral_updates.empty:
            user_features_df = user_features_df.merge(behavioral_updates, on="user_id", how="left")
            mask = user_features_df["top_category_realtime"].notna()
            user_features_df.loc[mask, "top_category"] = (
                user_features_df.loc[mask, "top_category_realtime"].astype("int32")
            )
            user_features_df.drop(columns=["top_category_realtime"], inplace=True)
            logging.info("Applied real-time top_category overrides for %d user(s)", mask.sum())

    item_entity_df = pd.DataFrame({
        "item_id": interactions_df["item_id"].unique(),
        "event_timestamp": pd.Timestamp.now(tz="UTC"),
    })
    item_features_df = store.get_historical_features(
        entity_df=item_entity_df,
        features=[
            "item_features:price",
            "item_features:category_l1",
            "item_features:category_l2",
            "item_features:item_gender",
        ],
    ).to_df().drop(columns=["event_timestamp"])

    merged_df = (
        interactions_df
        .merge(user_features_df[["user_id", "age", "gender", "top_category"]], on="user_id", how="left")
        .merge(item_features_df[["item_id", "price", "category_l1", "category_l2", "item_gender"]], on="item_id", how="left")
    )
    merged_df["top_category"] = merged_df["top_category"].fillna(-1).astype("int32")

    split_idx = int(0.7 * len(merged_df))
    train_df = merged_df.iloc[:split_idx].copy()
    val_df   = merged_df.iloc[split_idx:].copy()
    train_df["click"] = 1
    val_df["click"]   = 1

    day_date = pd.Timestamp.now().strftime("%Y%m%d")
    train_df.to_parquet(os.path.join(new_data_path, f"train_day_{day_date}.parquet"), index=False)
    val_df.to_parquet(os.path.join(new_data_path, f"valid_day_{day_date}.parquet"), index=False)
    logging.info("Saved feature-merged splits: train_day_%s.parquet, valid_day_%s.parquet", day_date, day_date)

    valid_files = glob.glob(os.path.join(old_merged_path, "valid_day_*.parquet"))
    train_files = glob.glob(os.path.join(old_merged_path, "train_day_*.parquet"))
    sampled_train_files = random.sample(train_files, min(2, len(train_files)))
    old_merged_df = pd.concat([pd.read_parquet(f) for f in valid_files + sampled_train_files], ignore_index=True)
    logging.info("Loaded %d replay rows (%d valid + %d train files)", len(old_merged_df), len(valid_files), len(sampled_train_files))

    replay_train_df = (
        pd.concat([train_df, old_merged_df], ignore_index=True)
        .sample(frac=1)
        .reset_index(drop=True)
    )
    logging.info("Replay train size: %d | Validation size: %d", len(replay_train_df), len(val_df))

    workflow = Workflow.load(workflow_path)
    train_preprocessed = workflow.transform(Dataset(replay_train_df))
    valid_preprocessed = workflow.transform(Dataset(val_df))

    os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "valid"), exist_ok=True)
    train_preprocessed.to_parquet(os.path.join(output_path, "train"))
    valid_preprocessed.to_parquet(os.path.join(output_path, "valid"))
    logging.info("Incremental preprocessing complete. Output written to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_data_path",     type=str, required=True)
    parser.add_argument("--old_merged_path",   type=str, required=True)
    parser.add_argument("--workflow_path",     type=str, required=True)
    parser.add_argument("--feast_repo_path",   type=str, required=True)
    parser.add_argument("--output_path",       type=str, required=True)
    parser.add_argument("--feast_s3_bucket",   type=str, default=None,
                        help="S3 bucket for behavioral updates (enables skew elimination)")
    parser.add_argument("--feast_aws_region",  type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')
    logging.info("Args: %s", args)

    run_incremental_preprocessing(
        new_data_path=args.new_data_path,
        old_merged_path=args.old_merged_path,
        workflow_path=args.workflow_path,
        feast_repo_path=args.feast_repo_path,
        output_path=args.output_path,
        feast_s3_bucket=args.feast_s3_bucket,
        feast_aws_region=args.feast_aws_region,
    )
