import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import tensorflow as tf
import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.schema.tags import Tags
from merlin.dataloader.ops.embeddings import EmbeddingOperator
import logging
import argparse
logging.disable(logging.WARNING)


class DenseContinuousProjection(tf.keras.layers.Layer):
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


def build_two_tower_model(loader_schema):
    query_schema = loader_schema.select_by_tag([Tags.USER])
    candidate_schema = loader_schema.select_by_tag([Tags.ITEM])

    query_cont_branch = DenseContinuousProjection(
        query_schema.select_by_tag([Tags.CONTINUOUS]).column_names,
        units=8,
    )
    candidate_cont_branch = DenseContinuousProjection(
        candidate_schema.select_by_tag(Tags.CONTINUOUS).column_names,
        units=8,
    )

    query_embeddings = mm.Embeddings(
        query_schema.select_by_tag(Tags.CATEGORICAL),
        infer_embedding_sizes=True,
    )
    candidate_embeddings = mm.Embeddings(
        candidate_schema.select_by_tag(Tags.CATEGORICAL),
        infer_embedding_sizes=True,
    )

    query_input = mm.InputBlockV2(
        query_schema,
        categorical=query_embeddings,
        continuous=query_cont_branch,
    )
    candidate_input = mm.InputBlockV2(
        candidate_schema,
        categorical=candidate_embeddings,
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

    model = mm.TwoTowerModelV2(
        query_tower=query_encoder,
        candidate_tower=candidate_encoder,
        schema=loader_schema,
    )

    return model, query_encoder, candidate_encoder


def finetune_query_tower(input_path, embeddings_path, checkpoint_path, output_path, epochs=1, lr=1e-4):
    train_data = Dataset(os.path.join(input_path, "train", "*.parquet"))
    valid_data = Dataset(os.path.join(input_path, "valid", "*.parquet"))

    image_embeddings = np.load(os.path.join(embeddings_path, "lookup_embeddings_image.npy"))
    description_embeddings = np.load(os.path.join(embeddings_path, "lookup_embeddings_text.npy"))

    schema = train_data.schema.select_by_tag([Tags.USER, Tags.ITEM])
    train_data.schema = schema
    valid_data.schema = schema

    train_loader = mm.Loader(
        train_data,
        batch_size=1024,
        transforms=[
            EmbeddingOperator(image_embeddings, lookup_key="item_id", embedding_name="item_image_embeddings"),
            EmbeddingOperator(description_embeddings, lookup_key="item_id", embedding_name="item_description_embeddings"),
        ],
    )
    valid_loader = mm.Loader(
        valid_data,
        batch_size=1024,
        transforms=[
            EmbeddingOperator(image_embeddings, lookup_key="item_id", embedding_name="item_image_embeddings"),
            EmbeddingOperator(description_embeddings, lookup_key="item_id", embedding_name="item_description_embeddings"),
        ],
    )

    loader_schema = train_loader.output_schema
    model, query_encoder, candidate_encoder = build_two_tower_model(loader_schema)

    sample_batch = next(iter(train_loader))
    x = sample_batch[0] if isinstance(sample_batch, tuple) else sample_batch
    model(x, training=False)

    model.load_weights(os.path.join(checkpoint_path, "two_tower_model", "model.savedmodel", "variables", "variables"))

    candidate_encoder.trainable = False

    logging.info(
        "Trainable params: %d | Frozen params: %d",
        sum(tf.size(v).numpy() for v in model.trainable_variables),
        sum(tf.size(v).numpy() for v in model.non_trainable_variables),
    )

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
        run_eagerly=False,
        metrics=[mm.RecallAt(10), mm.NDCGAt(10)],
    )
    model.fit(train_loader, validation_data=valid_loader, epochs=epochs)

    query_encoder.save(os.path.join(output_path, "query_tower", "model.savedmodel"))

    model.trainable = True
    model.save(os.path.join(output_path, "two_tower_model", "model.savedmodel"))

    logging.info("Query tower fine-tuned and saved to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        required=True,
                        help="Path to incrementally processed NVT data (contains train/ and valid/)")
    parser.add_argument("--embeddings_path",
                        type=str,
                        required=True,
                        help="Path to static lookup_embeddings_image.npy and lookup_embeddings_text.npy")
    parser.add_argument("--checkpoint_path",
                        type=str,
                        required=True,
                        help="Path containing two_tower_model/model.savedmodel")
    parser.add_argument("--output_path",
                        type=str,
                        required=True,
                        help="Path to write updated query_tower/ and two_tower_model/")
    parser.add_argument("--epochs",
                        type=int,
                        default=1,
                        help="Number of epochs for fine-tuning")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="Learning rate for fine-tuning")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')
    logging.info("Args: %s", args)

    finetune_query_tower(
        input_path=args.input_path,
        embeddings_path=args.embeddings_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        epochs=args.epochs,
        lr=args.lr,
    )
