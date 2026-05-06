import os
import shutil
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import nvtabular as nvt
from nvtabular.ops import ColumnSelector, Rename, Filter, Dropna, LambdaOp, Categorify, \
    TagAsUserFeatures, TagAsUserID, TagAsItemFeatures, TagAsItemID, AddMetadata, LogOp, Normalize, Bucketize
import numpy as np
from merlin.schema.tags import Tags
from merlin.dag.ops.subgraph import Subgraph
from merlin.systems.dag.ops.workflow import TransformWorkflow
import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.dataloader.ops.embeddings import EmbeddingOperator
import pandas as pd
import json
from datetime import datetime
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from sklearn.decomposition import PCA
import joblib, torch
from tqdm import tqdm
import cudf, cupy
import gc

import tensorflow as tf
import logging
import argparse
logging.disable(logging.WARNING)


def run_preprocessing(input_path, base_dir, train_days, valid_days):

    output_path = os.path.join(base_dir, "processed_nvt")
    os.makedirs(output_path, exist_ok=True)

    for subdir in ["user_subworkflow", "item_subworkflow", "context_subworkflow", "full_workflow"]:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)

    for subdir in ["for_feature_store", "lookup_embeddings"]:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    items = cudf.read_parquet(os.path.join(input_path, "items.parquet"))
    users = cudf.read_parquet(os.path.join(input_path, "users.parquet"))

    for i in range(train_days):
        day = cudf.read_parquet(os.path.join(input_path, f"day_{i:02d}.parquet"))
        day = day.merge(users, on="user_id", how="left").merge(items, on="item_id", how="left")
        day['click'] = 1
        day.to_parquet(os.path.join(input_path, f"train_day_{i:02d}.parquet"), index=False)
        del day

    for i in range(train_days, train_days + valid_days):
        day = cudf.read_parquet(os.path.join(input_path, f"day_{i:02d}.parquet"))
        day = day.merge(users, on="user_id", how="left").merge(items, on="item_id", how="left")
        day['click'] = 1
        day.to_parquet(os.path.join(input_path, f"valid_day_{i:02d}.parquet"), index=False)
        del day
    
    del users, items
    gc.collect()

    
    items = Dataset(os.path.join(input_path, "items.parquet"))
    item_cats = ["item_id", "category_l1", "category_l2", "item_gender"] >> Categorify(dtype="int32", out_path=os.path.join(output_path, "item_subworkflow", "categories"))
    
    item_subworkflow = nvt.Workflow(
            (item_cats["item_id"] >> TagAsItemID()) +
            (item_cats[["category_l1", "category_l2", "item_gender"]] >> TagAsItemFeatures()) +
            (
                ["price"]
                >> LogOp()
                >> Normalize(out_dtype=np.float32)
                >> Rename(postfix="_log_norm")
                >> TagAsItemFeatures()
            )
        )

    item_subworkflow.fit(items)
    item_subworkflow.save(os.path.join(output_path, "item_subworkflow"))

    IMG_DIR = os.path.join(input_path, "item_images")
    IMG_CACHE_PATH = os.path.join(base_dir, "lookup_embeddings", "lookup_embeddings_image.npy")
    DESC_CACHE_PATH = os.path.join(base_dir, "lookup_embeddings", "lookup_embeddings_text.npy")

    unique_item_ids = pd.read_parquet(os.path.join(output_path, "item_subworkflow", "categories", "unique.item_id.parquet"))
    if os.path.exists(IMG_CACHE_PATH):
        print("Loading cached embeddings...")
        image_embeddings = np.load(IMG_CACHE_PATH)
        logging.info(f"Loaded image embeddings from {IMG_CACHE_PATH}")
    else:
        logging.info("Computing image embeddings...")
        raw_to_cat = dict(zip(unique_item_ids["item_id"], unique_item_ids.index))
        max_cat_idx = unique_item_ids.index.max()

        raw_embeddings = np.zeros((max_cat_idx + 1, 512), dtype=np.float32)

        clip_model = CLIPModel.from_pretrained(os.path.join(input_path, "embedding_models/clip-vit-base-patch32"))
        clip_processor = CLIPProcessor.from_pretrained(os.path.join(input_path, "embedding_models/clip-vit-base-patch32"))


        raw_ids = sorted(unique_item_ids["item_id"].tolist())
        batch_size = 64
        for i in tqdm(range(0, len(raw_ids), batch_size), desc="Encoding images"):
            batch_raw_ids = raw_ids[i : i + batch_size]
            batch_images = []
            batch_cat_ids = []

            for raw_id in batch_raw_ids:
                img_path = os.path.join(IMG_DIR, f"{raw_id}.jpg")
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                    batch_cat_ids.append(raw_to_cat[raw_id])
            if batch_images:
                inputs = clip_processor(images=batch_images, return_tensors="pt", padding=True)
                with torch.no_grad():
                    feats = clip_model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                for idx, cat_id in enumerate(batch_cat_ids):
                    raw_embeddings[cat_id] = feats[idx].cpu().numpy()
        non_zero_mask = np.any(raw_embeddings != 0, axis=1)
        pca = PCA(n_components=64)
        pca.fit(raw_embeddings[non_zero_mask])

        image_embeddings = np.zeros((max_cat_idx + 1, 64), dtype=np.float32)
        image_embeddings[non_zero_mask] = pca.transform(raw_embeddings[non_zero_mask])
        np.save(IMG_CACHE_PATH, image_embeddings)
        joblib.dump(pca, os.path.join(base_dir, "lookup_embeddings", "img_pca_model.pkl"))
        logging.info(f"Saved image embeddings to {IMG_CACHE_PATH} and PCA model to {os.path.join(base_dir, 'lookup_embeddings', 'img_pca_model.pkl')}")

    if os.path.exists(DESC_CACHE_PATH):
        logging.info(f"Loading cached description embeddings from {DESC_CACHE_PATH}...")
        desc_embeddings = np.load(DESC_CACHE_PATH)
        logging.info(f"Loaded description embeddings from {DESC_CACHE_PATH} with shape {desc_embeddings.shape}")
    else:
        logging.info("generating new embeddings...")
        item_descs = pd.read_parquet(os.path.join(input_path, "item_descriptions.parquet"))
        raw_to_cat = dict(zip(unique_item_ids["item_id"], unique_item_ids.index))
        max_cat_idx = unique_item_ids.index.max()

        text_model = SentenceTransformer(os.path.join(input_path, "embedding_models/sentence-transformer-model"))
        raw_ids = item_descs["item_id"].tolist()
        descriptions = item_descs["product_description"].tolist()
        raw_embs = text_model.encode(descriptions, show_progress_bar=True) # (2465, 384)

        # dim 384 to 64 with PCA
        pca_desc = PCA(n_components=64)
        reduced = pca_desc.fit_transform(raw_embs)

        # place into array indexed by categorified ID
        desc_embeddings = np.zeros((max_cat_idx + 1, 64), dtype=np.float32)
        for i, raw_id in enumerate(raw_ids):
            cat_idx = raw_to_cat.get(raw_id)
            if cat_idx is not None:
                desc_embeddings[cat_idx] = reduced[i]
        np.save(DESC_CACHE_PATH, desc_embeddings)
        joblib.dump(pca_desc, os.path.join(base_dir, "lookup_embeddings", "desc_pca_model.pkl"))
        logging.info(f"Saved description embeddings to {DESC_CACHE_PATH} and PCA model to {os.path.join(base_dir, 'lookup_embeddings', 'desc_pca_model.pkl')}")



    #add datetime and created timestamp for both user and item tables
    users = pd.read_parquet(os.path.join(input_path, "users.parquet"))
    users['datetime'] = datetime.now()
    users['datetime'] = users['datetime'].astype('datetime64[ns]')
    users['created'] = datetime.now()
    users['created'] = users['created'].astype('datetime64[ns]')
    users.to_parquet(os.path.join(base_dir, "for_feature_store", "user_features.parquet"), index=False)

    #items
    items = pd.read_parquet(os.path.join(input_path, "items.parquet"))
    items['datetime'] = datetime.now()
    items['datetime'] = items['datetime'].astype('datetime64[ns]')
    items['created'] = datetime.now()
    items['created'] = items['created'].astype('datetime64[ns]')

    items.to_parquet(os.path.join(base_dir, "for_feature_store", "item_features.parquet"), index=False)

    #convert interactions data to NVTabular datasets
    train_raw_paths = [os.path.join(input_path, f"train_day_{i:02d}.parquet") for i in range(train_days)]
    valid_raw_paths = [os.path.join(input_path, f"valid_day_{i:02d}.parquet") for i in range(train_days, train_days + valid_days)]
    train_raw = Dataset(train_raw_paths)
    valid_raw = Dataset(valid_raw_paths)

    #Feature Engineering with NVTabular

    #USER subworkflow
    age_norm_op = Normalize(out_dtype=np.float32)
    user_ops = (
        (["user_id"] >> Categorify(dtype="int32") >> TagAsUserID()) +
        (["age"] >> age_norm_op >> Rename(postfix="_norm") >> TagAsUserFeatures()) +
        (["age"] >> Bucketize({"age": [18, 25, 35, 50, 65]}) >> Rename(postfix="_binned") >> Categorify(dtype="int32") >> TagAsUserFeatures()) +
        (["gender"] >> Categorify(dtype="int32") >> TagAsUserFeatures())
    )

    user_subworkflow = nvt.Workflow(user_ops)
    user_subworkflow.fit(train_raw)
    user_subworkflow.save(os.path.join(output_path, "user_subworkflow"))
    logging.info(f"Mean users age: {age_norm_op.means}, Std users age: {age_norm_op.stds}")

    #save the mean age for use as cold start defualt age later.
    age_stats = {
    "age_mean": int(age_norm_op.means["age"]),
    }
    with open(os.path.join(output_path, "age_stats.json"), "w") as f:
        json.dump(age_stats, f)

    #CONTEXT subworkflow
    timestamp_col = ["timestamp"]
    hour_of_day = (
        timestamp_col
        >> LambdaOp(lambda col: cudf.to_datetime(col, unit="s").dt.hour)
        >> Rename(name="hour_of_day")
    )
    hour_sine = (
        hour_of_day
        >> LambdaOp(lambda col: np.sin(2 * np.pi * col / 24.0).astype("float32"))
        >> Rename(name="hour_sine")
    )
    hour_cosine = (
        hour_of_day
        >> LambdaOp(lambda col: np.cos(2 * np.pi * col / 24.0).astype("float32"))
        >> Rename(name="hour_cosine")
    )
    day_of_week = (
        timestamp_col
        >> LambdaOp(lambda col: cudf.to_datetime(col, unit="s").dt.weekday)
        >> Rename(name="day_of_week")
    )
    day_of_week_sine = (
        day_of_week
        >> LambdaOp(lambda col: np.sin(2 * np.pi * col / 7.0).astype("float32"))
        >> Rename(name="day_of_week_sine")
    )
    day_of_week_cosine = (
        day_of_week
        >> LambdaOp(lambda col: np.cos(2 * np.pi * col / 7.0).astype("float32"))
        >> Rename(name="day_of_week_cosine")
    )
    time_features  = hour_sine + hour_cosine + day_of_week_sine + day_of_week_cosine

    context_ops = (
        (["device_type"] >> Categorify(dtype="int32")) +
        (time_features >> AddMetadata(tags=[Tags.CONTINUOUS]))
    )
    context_subworkflow = nvt.Workflow(context_ops)
    context_subworkflow.fit(train_raw)
    context_subworkflow.save(os.path.join(output_path, "context_subworkflow"))

    #COMPOSE full workflow
    subgraph_item = Subgraph(
        "item",
        ["item_id", "category_l1", "category_l2", "price", "item_gender"] >> TransformWorkflow(item_subworkflow)
    )
    subgraph_user = Subgraph(
        "user", 
        ["user_id", "age", "gender"] >> TransformWorkflow(user_subworkflow)
    )   
    subgraph_context = Subgraph(
        "context",
        ["device_type", "timestamp"] >> TransformWorkflow(context_subworkflow)
    )
    targets = ["click"] >> AddMetadata(tags=[Tags.BINARY_CLASSIFICATION, Tags.TARGET])
    outputs = subgraph_item + subgraph_user + subgraph_context + targets
    
    full_workflow = nvt.Workflow(outputs)
    full_workflow.fit(train_raw)
    full_workflow.save(os.path.join(output_path, "full_workflow"))

    
    #MASK some users and context features in train data with 5% probability 
    ANONYMOUS_USER = -1
    OOV_GENDER = -1
    OOV_DEVICE = -1
    masked_train_dir = os.path.join(input_path, "masked_train")
    os.makedirs(masked_train_dir, exist_ok=True)

    for i in range(train_days):
        day = cudf.read_parquet(os.path.join(input_path, f"train_day_{i:02d}.parquet"))
        n=len(day)
        user_mask = cupy.random.random(n) < 0.05
        day.loc[user_mask, "user_id"] = ANONYMOUS_USER
        day.loc[user_mask, "gender"] = OOV_GENDER

        device_mask = cupy.random.random(n) < 0.05
        day.loc[device_mask, "device_type"] = OOV_DEVICE
        day.to_parquet(os.path.join(masked_train_dir, f"train_day_{i:02d}.parquet"), index=False)
        del day
        gc.collect()
    
    masked_train_paths = [os.path.join(masked_train_dir, f"train_day_{i:02d}.parquet") for i in range(train_days)]
    masked_train_ds = Dataset(masked_train_paths)

    full_workflow.transform(masked_train_ds).to_parquet(os.path.join(output_path, "train"))
    full_workflow.transform(valid_raw).to_parquet(os.path.join(output_path, "valid"))

    logging.info("preprocessing complete")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        required=False,
                        default="/var/lib/data/raw_data",
                        help="path to input raw data files day_<day_number>.parquet, items.parquet, users.parquet, and item_images/")
    parser.add_argument("--base_dir",
                        type=str,
                        required=False,
                        default="/var/lib/data/processed_data",
                        help="base directory for all preprocessed outputs")
    parser.add_argument("--train_days",
                        type=int,
                        required=False,
                        default=9,
                        help="number of days to use for training set")
    parser.add_argument("--valid_days",
                        type=int,
                        required=False,
                        default=3,
                        help="number of days to use for validation set")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')
    logging.info(f"Args: {args}")

    run_preprocessing(input_path=args.input_path,
                      base_dir=args.base_dir,
                      train_days=args.train_days,
                      valid_days=args.valid_days)



















