"""
Model 5a: Feast Item Lookup (Python Backend) — In-Memory Cache
===============================================================
Loads all item features from Feast once at startup into numpy arrays
indexed by raw item_id. Each requestlookup is a pure numpy index (~0.1ms)
instead of a Feast online-store round trip (E.g., ~157ms for 300 items).

Input:  filtered_ids (TYPE_INT32, [-1]) — item IDs after seen-filter
Output: item_id      (TYPE_INT32, [-1])
        price        (TYPE_FP32,  [-1])
        category_l1  (TYPE_INT32, [-1])
        category_l2  (TYPE_INT32, [-1])
        item_gender  (TYPE_INT32, [-1])

Staleness: item features are refreshed only on Triton restart/reload.
Acceptable for static catalog attributes (category, gender, price tier).
"""

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        import feast
        import pathlib

        feast_repo = "/model/script/feast_repo/feature_repo"
        store = feast.FeatureStore(feast_repo)

        feature_refs = [
            "item_features:price",
            "item_features:category_l1",
            "item_features:category_l2",
            "item_features:item_gender",
        ]

        # Load the full item vocabulary from the NVT categories file
        # to know which raw item_ids exist in the catalog
        repository_path = pathlib.Path(args["model_repository"])
        if str(repository_path).endswith(".py"):
            repository_path = repository_path.parent.parent
        ver = args["model_version"]

        import pandas as pd
        unique_items_df = pd.read_parquet(
            repository_path / ver / "categories" / "unique.item_id.parquet"
        )
        all_raw_ids = unique_items_df["item_id"].dropna().astype(int).tolist()

        # Bulk-fetch all item features from Feast in one call
        entity_rows = [{"item_id": iid} for iid in all_raw_ids]
        feast_dict = store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        ).to_dict()

        # Build numpy arrays indexed by raw item_id for O(1) lookup
        max_id = max(all_raw_ids) + 1
        self._price       = np.zeros(max_id, dtype=np.float32)
        self._category_l1 = np.zeros(max_id, dtype=np.int32)
        self._category_l2 = np.zeros(max_id, dtype=np.int32)
        self._item_gender = np.zeros(max_id, dtype=np.int32)

        for i, iid in enumerate(feast_dict["item_id"]):
            if iid is None:
                continue
            iid = int(iid)
            self._price[iid]       = feast_dict["price"][i]       or 0.0
            self._category_l1[iid] = feast_dict["category_l1"][i] or 0
            self._category_l2[iid] = feast_dict["category_l2"][i] or 0
            self._item_gender[iid] = feast_dict["item_gender"][i]  or 0

        pb_utils.Logger.log_info(
            f"5a_feast_item ready: cached {len(all_raw_ids)} items (max_id={max_id - 1})"
        )

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
        item_ids = pb_utils.get_input_tensor_by_name(
            request, "filtered_ids"
        ).as_numpy().reshape(-1).astype(np.int32)

        safe_ids = np.clip(item_ids, 0, len(self._price) - 1)

        return pb_utils.InferenceResponse([
            pb_utils.Tensor("item_id",     item_ids),
            pb_utils.Tensor("price",       self._price[safe_ids]),
            pb_utils.Tensor("category_l1", self._category_l1[safe_ids]),
            pb_utils.Tensor("category_l2", self._category_l2[safe_ids]),
            pb_utils.Tensor("item_gender", self._item_gender[safe_ids]),
        ])
