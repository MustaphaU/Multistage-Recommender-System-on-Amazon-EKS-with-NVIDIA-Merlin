import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import glob
import random
import numpy as np
import pandas as pd
import feast
import nvtabular as nvt
from nvtabular import Workflow
from merlin.io.dataset import Dataset
import logging
import argparse
logging.disable(logging.WARNING)


def run_incremental_preprocessing(new_data_path, old_merged_path, workflow_path, feast_repo_path, output_path):

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
        features=["user_features:age", "user_features:gender"],
    ).to_df().drop(columns=["event_timestamp"])

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
        .merge(user_features_df[["user_id", "age", "gender"]], on="user_id", how="left")
        .merge(item_features_df[["item_id", "price", "category_l1", "category_l2", "item_gender"]], on="item_id", how="left")
    )

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
    parser.add_argument("--new_data_path",   type=str, required=True, help="Path to new day raw interactions (day_*.parquet)")
    parser.add_argument("--old_merged_path", type=str, required=True, help="Path to replay data (train_day_*.parquet + valid_day_*.parquet)")
    parser.add_argument("--workflow_path",   type=str, required=True, help="Path to fitted NVT full_workflow")
    parser.add_argument("--feast_repo_path", type=str, required=True, help="Path to Feast feature_repo directory")
    parser.add_argument("--output_path",     type=str, required=True, help="Path to write processed train/ and valid/")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')
    logging.info("Args: %s", args)

    run_incremental_preprocessing(
        new_data_path=args.new_data_path,
        old_merged_path=args.old_merged_path,
        workflow_path=args.workflow_path,
        feast_repo_path=args.feast_repo_path,
        output_path=args.output_path,
    )
