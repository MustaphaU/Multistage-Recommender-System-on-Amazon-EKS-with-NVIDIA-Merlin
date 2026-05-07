"""
Model 6c: SoftmaxSampling (Python Backend)
============================================
Weighted sampling without replacement via the exponential sort trick.

Reference:
    Efraimidis & Spirakis, "Weighted random sampling with a reservoir",
    Information Processing Letters 97(5), 2006.

Input:  item_id                         (TYPE_INT32, [-1])   — candidate IDs (N)
        click/binary_classification_task (TYPE_FP32, [-1, 1]) — DLRM scores  (N)
Output: ordered_ids    (TYPE_INT32, [-1])  — top-K item IDs
        ordered_scores (TYPE_FP32,  [-1])  — corresponding scores

Environment:
    RANKING_TOPK         — Number of items to return (default: 10)
    SOFTMAX_TEMPERATURE  — Temperature parameter (default: 1e-8)
"""

# import os
# import logging

# import numpy as np
# import triton_python_backend_utils as pb_utils

# logger = logging.getLogger("6c_softmax_sampling")

# TOPK = int(os.environ.get("RANKING_TOPK", "10"))
# TEMPERATURE = float(os.environ.get("SOFTMAX_TEMPERATURE", "1e-8"))


# class TritonPythonModel:
#     def initialize(self, args):
#         logger.info("6c_softmax_sampling ready  (topk=%d, temp=%.1e)", TOPK, TEMPERATURE)

#     def execute(self, requests):
#         responses = []
#         for request in requests:
#             try:
#                 responses.append(self._handle(request))
#             except Exception as exc:
#                 import traceback
#                 responses.append(
#                     pb_utils.InferenceResponse(
#                         error=pb_utils.TritonError(traceback.format_exc())
#                     )
#                 )
#         return responses

#     def _handle(self, request):
#         candidate_ids = pb_utils.get_input_tensor_by_name(
#             request, "item_id"
#         ).as_numpy().reshape(-1)

#         scores = pb_utils.get_input_tensor_by_name(
#             request, "click/binary_classification_task"
#         ).as_numpy().reshape(-1)

#         num_items = len(candidate_ids)
#         topk = min(TOPK, num_items)

#         weights = np.exp(TEMPERATURE * scores) / np.sum(scores)
#         exponentials = -np.log(np.random.uniform(0, 1, size=num_items))
#         exponentials /= weights

#         sorted_idx = np.argsort(exponentials)
#         top_ids = candidate_ids[sorted_idx][:topk]
#         top_scores = scores[sorted_idx][:topk]

#         return pb_utils.InferenceResponse([
#             pb_utils.Tensor("ordered_ids", top_ids.astype(np.int32)),
#             pb_utils.Tensor("ordered_scores", top_scores.astype(np.float32)),
#         ])



import numpy as np
import triton_python_backend_utils as pb_utils

TEMPERATURE = 20.0


class TritonPythonModel:
    def initialize(self, args):
        import json
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        self.topk = int(params.get("RANKING_TOPK", {}).get("string_value", "10"))
        self.diversity_mode = params.get("DIVERSITY_MODE", {}).get("string_value", "false").lower() == "true"

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
        candidate_ids = pb_utils.get_input_tensor_by_name(
            request, "item_id"
        ).as_numpy().reshape(-1)

        scores = pb_utils.get_input_tensor_by_name(
            request, "click/binary_classification_task"
        ).as_numpy().reshape(-1)

        #pb_utils.Logger.log_warn(f"candidates are received by softmax model: {candidate_ids.tolist()}")
        num_items = len(candidate_ids)
        topk = min(self.topk, num_items)

        weights = np.exp(TEMPERATURE * scores) / np.sum(scores)
        if not self.diversity_mode:
            # Deterministic: just take top-K by raw score (greedy)
            sorted_idx = np.argsort(-scores)
            top_ids = candidate_ids[sorted_idx][:topk]
            top_scores = scores[sorted_idx][:topk]
            return pb_utils.InferenceResponse([
                pb_utils.Tensor("ordered_ids", top_ids.astype(np.int32)),
                pb_utils.Tensor("ordered_scores", top_scores.astype(np.float32)),
            ])
        exponentials = -np.log(np.random.uniform(0, 1, size=num_items))
        exponentials /= weights

        sorted_idx = np.argsort(exponentials)
        top_ids = candidate_ids[sorted_idx][:topk]
        top_scores = scores[sorted_idx][:topk]

        return pb_utils.InferenceResponse([
            pb_utils.Tensor("ordered_ids", top_ids.astype(np.int32)),
            pb_utils.Tensor("ordered_scores", top_scores.astype(np.float32)),
        ])
