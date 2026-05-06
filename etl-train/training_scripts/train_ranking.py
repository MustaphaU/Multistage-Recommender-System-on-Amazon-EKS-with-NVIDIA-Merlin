import os
import shutil
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import nvtabular as nvt
from nvtabular.ops import ColumnSelector, Rename, Filter, Dropna, LambdaOp, Categorify, \
    TagAsUserFeatures, TagAsUserID, TagAsItemFeatures, TagAsItemID, AddMetadata, LogOp, Normalize, Bucketize
from merlin.schema.tags import Tags
from merlin.dag import BaseOperator
from merlin.table import TensorTable
from merlin.dag.ops.subgraph import Subgraph
from merlin.systems.dag.ops.workflow import TransformWorkflow
import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.dataloader.ops.embeddings import EmbeddingOperator
import pandas as pd
from datetime import datetime
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from sklearn.decomposition import PCA
import joblib, torch
from tqdm import tqdm

import tensorflow as tf
import logging
import argparse
logging.disable(logging.WARNING)
from merlin.models.tf.transforms.negative_sampling import InBatchNegatives
from merlin.models.tf.core.combinators import ParallelBlock

def train_dlrm(input_path, output_path):
    train_data = Dataset(os.path.join(input_path, "processed_nvt", "train", "*.parquet"))
    valid_data = Dataset(os.path.join(input_path, "processed_nvt", "valid", "*.parquet"))
    
    image_embeddings = np.load(os.path.join(input_path, "lookup_embeddings", "lookup_embeddings_image.npy"))
    description_embeddings = np.load(os.path.join(input_path, "lookup_embeddings", "lookup_embeddings_text.npy"))

    train_loader = mm.Loader(
        train_data,
        batch_size = 16 * 1024,
        shuffle=True,
        transforms=[
            EmbeddingOperator(
                image_embeddings,
                lookup_key="item_id",
                embedding_name="item_image_embeddings"
            ),
            EmbeddingOperator(
                description_embeddings,
                lookup_key="item_id",
                embedding_name="item_description_embeddings"
            ),
        ],
    )

    valid_loader =  mm.Loader(
        valid_data,
        batch_size=16*1024,
        transforms=[
            EmbeddingOperator(
                image_embeddings,
                lookup_key="item_id",
                embedding_name="item_image_embeddings"
            ),
            EmbeddingOperator(
                description_embeddings,
                lookup_key="item_id",
                embedding_name="item_description_embeddings"
            ),
        ],
    )

    target_column = train_data.schema.select_by_tag(Tags.TARGET).column_names[0]
    ranking_schema = train_loader.output_schema

    add_negatives = InBatchNegatives(ranking_schema, n_per_positive=5, prep_features=True, run_when_testing=False)

    ranking_embeddings = ParallelBlock(
        {
            "categorical": mm.Embeddings(
                ranking_schema.select_by_tag(Tags.CATEGORICAL),
                infer_embedding_sizes=False,
                dim=64,
            ),
            "pretrained_embeddings": mm.PretrainedEmbeddings(
                ranking_schema.select_by_tag(Tags.EMBEDDING),
                output_dims=64
            ),
        },
        aggregation=None,
        is_input=True,
        schema=ranking_schema
    )

    ranking_model = mm.DLRMModel(
        ranking_schema,
        embeddings=ranking_embeddings,
        bottom_block=mm.MLPBlock([128, 64]),
        top_block=mm.MLPBlock([128, 64, 32]),
        prediction_tasks=mm.BinaryClassificationTask(target_column),
    )

    ranking_model.compile(
        optimizer="adam",
        run_eagerly=True,
        metrics=[tf.keras.metrics.AUC()]
    )

    ranking_model.fit(
        train_loader,
        validation_data=valid_loader,
        epochs=2,
        pre=add_negatives
    )

    logging.info("running evaluation on the validation set with inbatch negatives...")
    add_negatives = InBatchNegatives(ranking_schema, n_per_positive=5, prep_features=True, run_when_testing=True)
    ranking_model.evaluate(valid_loader, pre=add_negatives)

    ranking_model.save(os.path.join(output_path, "dlrm", "model.savedmodel"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                         type=str, 
                         default="/var/lib/data/processed")
    parser.add_argument("--output_path",
                         type=str,
                         default="/var/lib/data/models")
    args = parser.parse_args()

    train_dlrm(args.input_path, args.output_path)









