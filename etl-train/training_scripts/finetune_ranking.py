import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import tensorflow as tf
import merlin.models.tf as mm
from merlin.io.dataset import Dataset
from merlin.schema.tags import Tags
from merlin.dataloader.ops.embeddings import EmbeddingOperator
from merlin.models.tf.transforms.negative_sampling import InBatchNegatives
from merlin.models.tf.core.combinators import ParallelBlock
import logging
import argparse
logging.disable(logging.WARNING)


def build_dlrm(ranking_schema, target_column):

    ranking_embeddings = ParallelBlock(
        {
            "categorical": mm.Embeddings(
                ranking_schema.select_by_tag(Tags.CATEGORICAL),
                infer_embedding_sizes=False,
                dim=64,
            ),
            "pretrained_embeddings": mm.PretrainedEmbeddings(
                ranking_schema.select_by_tag(Tags.EMBEDDING),
                output_dims=64,
            ),
        },
        aggregation=None,
        is_input=True,
        schema=ranking_schema,
    )

    model = mm.DLRMModel(
        ranking_schema,
        embeddings=ranking_embeddings,
        bottom_block=mm.MLPBlock([128, 64]),
        top_block=mm.MLPBlock([128, 64, 32]),
        prediction_tasks=mm.BinaryClassificationTask(target_column),
    )

    return model


def finetune_dlrm(input_path, embeddings_path, checkpoint_path, output_path, epochs=1, lr=1e-4):
    train_data = Dataset(os.path.join(input_path, "train", "*.parquet"))
    valid_data = Dataset(os.path.join(input_path, "valid", "*.parquet"))

    image_embeddings = np.load(os.path.join(embeddings_path, "lookup_embeddings_image.npy"))
    description_embeddings = np.load(os.path.join(embeddings_path, "lookup_embeddings_text.npy"))

    train_loader = mm.Loader(
        train_data,
        batch_size=16 * 1024,
        shuffle=True,
        transforms=[
            EmbeddingOperator(image_embeddings, lookup_key="item_id", embedding_name="item_image_embeddings"),
            EmbeddingOperator(description_embeddings, lookup_key="item_id", embedding_name="item_description_embeddings"),
        ],
    )
    valid_loader = mm.Loader(
        valid_data,
        batch_size=16 * 1024,
        transforms=[
            EmbeddingOperator(image_embeddings, lookup_key="item_id", embedding_name="item_image_embeddings"),
            EmbeddingOperator(description_embeddings, lookup_key="item_id", embedding_name="item_description_embeddings"),
        ],
    )

    target_column = train_data.schema.select_by_tag(Tags.TARGET).column_names[0]
    ranking_schema = train_loader.output_schema
    model = build_dlrm(ranking_schema, target_column)

    sample_batch = next(iter(train_loader))
    x = sample_batch[0] if isinstance(sample_batch, tuple) else sample_batch
    model(x, training=False)

    model.load_weights(os.path.join(checkpoint_path, "dlrm", "model.savedmodel", "variables", "variables"))

    logging.info(
        "Trainable params: %d | Frozen params: %d",
        sum(tf.size(v).numpy() for v in model.trainable_variables),
        sum(tf.size(v).numpy() for v in model.non_trainable_variables),
    )

    add_negatives_train = InBatchNegatives(ranking_schema, n_per_positive=5, prep_features=True, run_when_testing=False)
    add_negatives_eval = InBatchNegatives(ranking_schema, n_per_positive=5, prep_features=True, run_when_testing=True)

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
        run_eagerly=True,
        metrics=[tf.keras.metrics.AUC()],
    )
    model.fit(train_loader, validation_data=valid_loader, epochs=epochs, pre=add_negatives_train)

    eval_results = model.evaluate(valid_loader, pre=add_negatives_eval, return_dict=True)
    logging.info("Validation (with in-batch negatives): %s", eval_results)

    model.save(os.path.join(output_path, "dlrm", "model.savedmodel"))
    logging.info("DLRM fine-tuned and saved to %s", output_path)


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
                        help="Path containing dlrm/model.savedmodel")
    parser.add_argument("--output_path",
                        type=str,
                        required=True,
                        help="Path to write updated dlrm/")
    parser.add_argument("--epochs",
                        type=int,
                        default=1)
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')
    logging.info("Args: %s", args)

    finetune_dlrm(
        input_path=args.input_path,
        embeddings_path=args.embeddings_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        epochs=args.epochs,
        lr=args.lr,
    )
