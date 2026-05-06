"""
Model 5a: Feast Item Lookup (Python Backend)
==============================================
Feast online store → raw item features.

Input:  filtered_ids (TYPE_INT32, [-1]) — item IDs after seen-filter
Output: item_id (TYPE_INT32), price (TYPE_FP32),
        category_l1 (TYPE_INT32), category_l2 (TYPE_INT32), item_gender (TYPE_INT32)

Environment:
    FEAST_REPO_PATH — Path to Feast feature repository

Update cadence: Feast online store refreshed by hourly materialization job.
"""

import os
import logging

import numpy as np
import triton_python_backend_utils as pb_utils

# logger = logging.getLogger("5a_feast_item")


class TritonPythonModel:
    def initialize(self, args):
        import feast

        feast_repo = "/model/script/feast_repo/feature_repo"
        self.store = feast.FeatureStore(feast_repo)
        self.feature_refs = [
            "item_features:price",
            "item_features:category_l1",
            "item_features:category_l2",
            "item_features:item_gender",
        ]
        # logger.info("5a_feast_item ready  (feast=%s)", feast_repo)

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
        filtered_ids = pb_utils.get_input_tensor_by_name(
            request, "filtered_ids"
        ).as_numpy().reshape(-1)

        pb_utils.Logger.log_warn(f"filtered item ids received by item Feast lookup model: {filtered_ids.tolist()}")

        entity_rows = [{"item_id": int(iid)} for iid in filtered_ids]
        feast_dict = self.store.get_online_features(
            features=self.feature_refs,
            entity_rows=entity_rows,
        ).to_dict()

        return pb_utils.InferenceResponse([
            pb_utils.Tensor("item_id",     np.array(feast_dict["item_id"],     dtype=np.int32)),
            pb_utils.Tensor("price",       np.array(feast_dict["price"],       dtype=np.float32)),
            pb_utils.Tensor("category_l1", np.array(feast_dict["category_l1"], dtype=np.int32)),
            pb_utils.Tensor("category_l2", np.array(feast_dict["category_l2"], dtype=np.int32)),
            pb_utils.Tensor("item_gender", np.array(feast_dict["item_gender"], dtype=np.int32)),
        ])
