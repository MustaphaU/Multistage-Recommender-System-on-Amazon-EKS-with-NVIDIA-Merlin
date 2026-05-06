import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import nvtabular as nvt
from nvtabular.ops import ColumnSelector, Rename, Filter, Dropna, LambdaOp, Categorify, \
    TagAsUserFeatures, TagAsUserID, TagAsItemFeatures, TagAsItemID, AddMetadata, LogOp, Normalize, Bucketize
from merlin.schema.tags import Tags
from merlin.systems.dag.ops.faiss import setup_faiss
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
import tensorflow as tf
import logging
import argparse
logging.disable(logging.WARNING)


#Helper classes and functions for embedding lookup
def _get_cupy_module():
    try:
        import cupy as cupy
    except ImportError:
        return None
    return cupy


def _is_cupy_array(value):
    cupy = _get_cupy_module()
    return cupy is not None and isinstance(value, cupy.ndarray)


class LookupEmbeddings(BaseOperator):
    def __init__(
        self,
        image_matrix=None,
        text_matrix=None,
        embed_schema=None,
        id_col="item_id",
        image_col="item_image_embeddings",
        text_col="item_description_embeddings",
        artifact_prefix="lookup_embeddings",
    ):
        super().__init__()
        self.embed_schema = embed_schema
        self.id_col = id_col
        self.image_col = image_col
        self.text_col = text_col
        self.image_artifact_name = f"{artifact_prefix}_image.npy"
        self.text_artifact_name = f"{artifact_prefix}_text.npy"
        self.image_matrix_np = None
        self.text_matrix_np = None
        self.image_matrix_cp = None
        self.text_matrix_cp = None

        if image_matrix is not None and text_matrix is not None:
            self._set_matrices(image_matrix, text_matrix)

    def _set_matrices(self, image_matrix, text_matrix):
        self.image_matrix_np = np.asarray(image_matrix, dtype=np.float32)
        self.text_matrix_np = np.asarray(text_matrix, dtype=np.float32)
        self.image_matrix_cp = None
        self.text_matrix_cp = None

    def _ensure_cpu_matrices(self):
        if self.image_matrix_np is None or self.text_matrix_np is None:
            raise RuntimeError("Embedding matrices are not loaded")

    def _ensure_gpu_matrices(self):
        cupy = _get_cupy_module()
        if cupy is None:
            raise RuntimeError("cupy is required for GPU embedding lookup")
        self._ensure_cpu_matrices()
        if self.image_matrix_cp is None:
            self.image_matrix_cp = cupy.asarray(self.image_matrix_np)
        if self.text_matrix_cp is None:
            self.text_matrix_cp = cupy.asarray(self.text_matrix_np)
        return cupy

    def save_artifacts(self, artifact_path):
        self._ensure_cpu_matrices()
        np.save(os.path.join(artifact_path, self.image_artifact_name), self.image_matrix_np)
        np.save(os.path.join(artifact_path, self.text_artifact_name), self.text_matrix_np)

    def load_artifacts(self, artifact_path):
        image_path = os.path.join(artifact_path, self.image_artifact_name)
        text_path = os.path.join(artifact_path, self.text_artifact_name)
        if os.path.exists(image_path) and os.path.exists(text_path):
            self._set_matrices(np.load(image_path), np.load(text_path))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["image_matrix_cp"] = None
        state["text_matrix_cp"] = None
        state["image_matrix_np"] = None
        state["text_matrix_np"] = None
        return state

    def compute_output_schema(self, input_schema, col_selector, prev_output_schema=None):
        base = prev_output_schema if prev_output_schema is not None else input_schema
        return base + self.embed_schema

    def _lookup_gpu(self, ids):
        cupy = self._ensure_gpu_matrices()
        ids_cp = cupy.asarray(ids).reshape(-1).astype(cupy.int32)
        return (
            self.image_matrix_cp[ids_cp].astype(cupy.float32),
            self.text_matrix_cp[ids_cp].astype(cupy.float32),
        )

    def _lookup_cpu(self, ids):
        self._ensure_cpu_matrices()
        ids_np = np.asarray(ids).reshape(-1).astype(np.int32)
        return (
            self.image_matrix_np[ids_np].astype(np.float32),
            self.text_matrix_np[ids_np].astype(np.float32),
        )

    def _to_tensor_compatible(self, column):
        if hasattr(column, "to_cupy"):
            return column.to_cupy()
        if _is_cupy_array(column):
            return column
        if hasattr(column, "values") and _is_cupy_array(column.values):
            return column.values
        if isinstance(column, np.ndarray):
            return column
        if hasattr(column, "to_numpy"):
            return column.to_numpy()
        return column

    def transform(self, col_selector, transformable):
        cols = transformable.to_dict() if isinstance(transformable, TensorTable) else dict(transformable)
        ids = cols[self.id_col]

        if hasattr(ids, "to_cupy"):
            image_embs, text_embs = self._lookup_gpu(ids.to_cupy())
        elif _is_cupy_array(ids):
            image_embs, text_embs = self._lookup_gpu(ids)
        elif hasattr(ids, "values") and _is_cupy_array(ids.values):
            image_embs, text_embs = self._lookup_gpu(ids.values)
        elif hasattr(ids, "to_numpy"):
            image_embs, text_embs = self._lookup_cpu(ids.to_numpy())
        else:
            image_embs, text_embs = self._lookup_cpu(ids)

        normalized_cols = {
            name: self._to_tensor_compatible(column)
            for name, column in cols.items()
        }
        normalized_cols[self.image_col] = image_embs
        normalized_cols[self.text_col] = text_embs
        return TensorTable(normalized_cols)


class ToHostArrays(BaseOperator):
    def compute_output_schema(self, input_schema, col_selector, prev_output_schema=None):
        return prev_output_schema if prev_output_schema is not None else input_schema

    def _to_host(self, column):
        cupy = _get_cupy_module()
        if hasattr(column, "to_numpy"):
            return column.to_numpy()
        if hasattr(column, "to_cupy") and cupy is not None:
            return cupy.asnumpy(column.to_cupy())
        if cupy is not None and isinstance(column, cupy.ndarray):
            return cupy.asnumpy(column)
        if hasattr(column, "values") and cupy is not None and isinstance(column.values, cupy.ndarray):
            return cupy.asnumpy(column.values)
        if isinstance(column, np.ndarray):
            return column
        return np.asarray(column)

    def transform(self, col_selector, transformable):
        cols = transformable.to_dict() if isinstance(transformable, TensorTable) else dict(transformable)
        return TensorTable({name: self._to_host(column) for name, column in cols.items()})


def train_twotower_and_setup_faiss(input_path, output_path):
    
    train_data = Dataset(os.path.join(input_path, "processed_nvt", "train", "*.parquet"))
    valid_data = Dataset(os.path.join(input_path, "processed_nvt", "valid", "*.parquet"))

    image_embeddings = np.load(os.path.join(input_path, "lookup_embeddings", "lookup_embeddings_image.npy"))
    description_embeddings = np.load(os.path.join(input_path, "lookup_embeddings", "lookup_embeddings_text.npy"))

    schema = train_data.schema.select_by_tag([Tags.USER, Tags.ITEM])
    train_data.schema = schema
    valid_data.schema = schema

    train_loader = mm.Loader(
        train_data,
        batch_size=1024,
        transforms=[
            EmbeddingOperator(image_embeddings, lookup_key="item_id", embedding_name="item_image_embeddings"),
            EmbeddingOperator(description_embeddings, lookup_key="item_id", embedding_name="item_description_embeddings")
        ],
    )
    valid_loader = mm.Loader(
        valid_data,
        batch_size=1024,
        transforms=[
            EmbeddingOperator(image_embeddings, lookup_key="item_id", embedding_name="item_image_embeddings"),
            EmbeddingOperator(description_embeddings, lookup_key="item_id", embedding_name="item_description_embeddings")
        ],
    )

    loader_schema = train_loader.output_schema
    query_schema = loader_schema.select_by_tag([Tags.USER])
    candidate_schema = loader_schema.select_by_tag([Tags.ITEM])

    #build the model

    class DenseContinuousProjection(tf.keras.layers.Layer):
        """
        `DenseContinuousProjection` is a small custom layer that converts continuous features from shape [batch] to [batch, 1],
        then projects them through a Dense layer to a learned vector like [batch, 8].
         We used it as the `continuous=` branch inside each InputBlockV2 so continuous features could be concatenated with categorical embeddings and pretrained item embeddings in the two-tower model.
        """
        def __init__(self, feature_names, units, **kwargs):
            super().__init__(**kwargs)
            self.feature_names = list(feature_names)
            self.projection = tf.keras.layers.Dense(units)

        def call(self, inputs):
            tensors = [
                tf.expand_dims(tf.cast(inputs[name], tf.float32), axis=-1)
                for name in self.feature_names
            ]
            continuous_tensor = tensors[0] if len(tensors) == 1 else tf.concat(tensors, axis=-1)
            return self.projection(continuous_tensor)

    query_cont_branch = DenseContinuousProjection(
        query_schema.select_by_tag([Tags.CONTINUOUS]).column_names,
        units=8,
    )
    candidate_cont_branch = DenseContinuousProjection(
        candidate_schema.select_by_tag(Tags.CONTINUOUS).column_names,
        units=8,
    )

    query_input = mm.InputBlockV2(
        query_schema,
        categorical=mm.Embeddings(
            query_schema.select_by_tag(Tags.CATEGORICAL),
            infer_embedding_sizes=True,
        ),
        continuous=query_cont_branch,
    )

    candidate_input = mm.InputBlockV2(
        candidate_schema,
        categorical=mm.Embeddings(
            candidate_schema.select_by_tag(Tags.CATEGORICAL),
            infer_embedding_sizes=True,
        ),
        continuous=candidate_cont_branch,
        pretrained_embeddings=mm.PretrainedEmbeddings(
            candidate_schema.select_by_tag(Tags.EMBEDDING)
        ),
    )

    query_encoder = mm.Encoder(
        query_input,
        mm.MLPBlock([128, 64], no_activation_last_layer=True),
        prep_features=False,
    )

    candidate_encoder = mm.Encoder(
        candidate_input,
        mm.MLPBlock([128, 64], no_activation_last_layer=True),
        prep_features=False,
    )
    model_two_tower = mm.TwoTowerModelV2(
        query_tower=query_encoder,
        candidate_tower=candidate_encoder,
        schema=loader_schema,
    )

    model_two_tower.compile(
        optimizer="adam",
        run_eagerly=False,
        metrics=[mm.RecallAt(10), mm.NDCGAt(10)],
    )
    model_two_tower.fit(train_loader, validation_data=valid_loader, epochs=2)

    logging.info("Training complete, saving the query tower...")

    query_tower = model_two_tower.query_encoder
    query_tower.save(os.path.join(output_path, "query_tower", "model.savedmodel"))

    model_two_tower.save(os.path.join(output_path, "two_tower_model", "model.savedmodel"))

    logging.info("Computing item embeddings...")
    candidate_encoder = model_two_tower.candidate_encoder
    embed_schema = candidate_encoder.schema.select_by_tag(Tags.EMBEDDING)

    raw_item_ds = Dataset(os.path.join(input_path, "for_feature_store", "item_features.parquet"))
    item_subworkflow = nvt.Workflow.load(os.path.join(input_path, "processed_nvt", "item_subworkflow"))

    processed_item_ds = item_subworkflow.transform(raw_item_ds)
    id_mapping = pd.read_parquet(os.path.join(input_path, "processed_nvt", "item_subworkflow", "categories", "unique.item_id.parquet")).reset_index()
    id_mapping.rename(columns={"index": "encoded_item_id"}, inplace=True)

    item_loader = mm.Loader(
        processed_item_ds,
        batch_size=8192,
        shuffle=False,
        transforms=[
            LookupEmbeddings(
                image_matrix=image_embeddings,
                text_matrix=description_embeddings,
                embed_schema=embed_schema,
                id_col="item_id",
                image_col="item_image_embeddings",
                text_col="item_description_embeddings",
            )
        ],
    )

    item_embs = model_two_tower.candidate_embeddings(item_loader, index="item_id", batch_size=8192)
    item_embs_df = item_embs.to_ddf().compute().to_pandas()
    item_embs_df = item_embs_df.reset_index().rename(columns={"item_id": "encoded_item_id"})

    merged_df = item_embs_df.merge(id_mapping[["encoded_item_id", "item_id"]], on="encoded_item_id", how="left")
    emb_cols = [c for c in merged_df.columns if c not in ["encoded_item_id", "item_id"]]

    final_embeddings_df = pd.DataFrame({
        "item_id": merged_df["item_id"].astype(int),
        "output_1": list(merged_df[emb_cols].to_numpy(dtype=np.float32))
    })

    final_embeddings_df.to_parquet(os.path.join(output_path, "item_embeddings.parquet"), index=False)

    logging.info("item_embeddings.parquet saved to %s", os.path.join(output_path, "item_embeddings.parquet"))

    os.makedirs(os.path.join(output_path, "faiss_index"), exist_ok=True)
    setup_faiss(item_vector=final_embeddings_df, output_path=os.path.join(output_path, "faiss_index", "index.faiss"), item_id_column="item_id", embedding_column="output_1")
    logging.info("FAISS index built and saved to %s", os.path.join(output_path, "faiss_index", "index.faiss"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        default="/var/lib/data/processed_data",
                        help="Path to the processed data directory")
    parser.add_argument("--output_path",
                        type=str,
                        default="/var/lib/data/models",
                        help="Path to the output models directory")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')
    logging.info(f"Args: {args}")

    train_twotower_and_setup_faiss(args.input_path, args.output_path)
