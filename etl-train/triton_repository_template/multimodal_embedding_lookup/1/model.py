"""
Model 5c: Embedding Lookup (Python Backend)
=============================================
Pre-computed item embedding matrix lookup by item_id.

Input:  item_id (TYPE_INT32, [-1]) — NVT-transformed item IDs
Output: item_image_embeddings       (TYPE_FP32, [-1, 64])
        item_description_embeddings (TYPE_FP32, [-1, 64])

Artifacts:
    1/lookup_embeddings_image.npy  — (n_items × 64) image embeddings
    1/lookup_embeddings_text.npy   — (n_items × 64) text embeddings

Update cadence: Rebuilt when item encoder is retrained (weekly).
"""

import pathlib
import logging

import numpy as np
import triton_python_backend_utils as pb_utils

# logger = logging.getLogger("5c_embedding_lookup")


class TritonPythonModel:
    def initialize(self, args):
        repository_path = pathlib.Path(args["model_repository"])
        if str(repository_path).endswith(".py"):
            repository_path = repository_path.parent.parent
        ver = args["model_version"]
        artifact_dir = repository_path / ver

        self.image_embeddings = np.load(str(artifact_dir / "lookup_embeddings_image.npy"))
        self.text_embeddings = np.load(str(artifact_dir / "lookup_embeddings_text.npy"))
        # logger.info(
        #     "5c_embedding_lookup ready  image=%s  text=%s",
        #     self.image_embeddings.shape, self.text_embeddings.shape,
        # )

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                responses.append(self._handle(request))
            except Exception as exc:
                import traceback
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(traceback.format_exc())
                    )
                )
        return responses

    def _handle(self, request):
        item_ids = pb_utils.get_input_tensor_by_name(request, "item_id").as_numpy().reshape(-1)

        pb_utils.Logger.log_warn(f"item_ids received by embedding lookup model: {item_ids.tolist()}")
        n_embed = self.image_embeddings.shape[0]
        safe_ids = np.clip(item_ids, 0, n_embed - 1) #not really necessary as any new item should have been embedded already and the workflow should have been updated.

        pb_utils.Logger.log_warn(f"safe_ids used for embedding lookup: {safe_ids.tolist()}")
        image_emb = self.image_embeddings[safe_ids].astype(np.float32)
        text_emb = self.text_embeddings[safe_ids].astype(np.float32)

        return pb_utils.InferenceResponse([
            pb_utils.Tensor("item_image_embeddings", image_emb),
            pb_utils.Tensor("item_description_embeddings", text_emb),
        ])
