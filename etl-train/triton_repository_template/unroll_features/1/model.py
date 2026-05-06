"""
Model 6a: UnrollFeatures (Python Backend)
==========================================
Cross-join 1 user × N items into N (user, item, context) pairs.

Tiles each scalar user feature and context feature N times to match
item array length, producing aligned arrays ready for the DLRM ranking model.

Input:  4 user features (scalar) + 5 context features (scalar) + 7 item features (N items)
Output: 4 user features (N each) + 5 context features (N each) + 7 item features (N each)

No artifacts. Stateless numpy reshaping.
"""

import logging

import numpy as np
import triton_python_backend_utils as pb_utils

logger = logging.getLogger("6a_unroll_features")

USER_FEATURES = ["user_id", "age_norm", "age_binned", "gender"]
CONTEXT_FEATURES = ["device_type", "hour_sine", "hour_cosine", "day_of_week_sine", "day_of_week_cosine"]
ITEM_FEATURES = [
    "item_id", "category_l1", "category_l2", "item_gender",
    "price_log_norm", "item_image_embeddings", "item_description_embeddings",
]


class TritonPythonModel:
    def initialize(self, args):
        logger.info("6a_unroll_features ready")

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
        tensors = {}
        for name in USER_FEATURES + CONTEXT_FEATURES + ITEM_FEATURES:
            tensors[name] = pb_utils.get_input_tensor_by_name(request, name).as_numpy()

        n_items = tensors["item_id"].reshape(-1).shape[0]

        # Tile user features: (1,) → (N,)
        for uf in USER_FEATURES:
            val = tensors[uf].reshape(-1)
            tensors[uf] = np.tile(val, n_items)

        # Tile context features: (1,) → (N,)
        for cf in CONTEXT_FEATURES:
            val = tensors[cf].reshape(-1)
            tensors[cf] = np.tile(val, n_items)

        # Reshape item features
        for feat in ITEM_FEATURES:
            arr = tensors[feat]
            if feat.endswith("_embeddings"):
                tensors[feat] = arr.reshape(n_items, -1)  # (N, 64)
            else:
                tensors[feat] = arr.reshape(-1)            # (N,)

        all_features = USER_FEATURES + CONTEXT_FEATURES + ITEM_FEATURES
        outputs = [pb_utils.Tensor(name, tensors[name]) for name in all_features]
        return pb_utils.InferenceResponse(outputs)
