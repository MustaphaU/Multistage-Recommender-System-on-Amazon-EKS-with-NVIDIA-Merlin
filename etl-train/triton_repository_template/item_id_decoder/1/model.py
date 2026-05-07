"""
Model 7: ID Decoder (Python Backend)
======================================
Final pipeline stage — converts NVT-encoded item indices back to
original item IDs using the Categorify mapping from the item NVT workflow.

Input:  ordered_ids    (TYPE_INT32, [-1]) — NVT-encoded item indices
        ordered_scores (TYPE_FP32,  [-1]) — passthrough scores
Output: ordered_ids    (TYPE_INT32, [-1]) — original item IDs
        ordered_scores (TYPE_FP32,  [-1]) — passthrough scores

Artifacts:
    1/categories/unique.item_id.parquet — NVT Categorify mapping

Update cadence: updated whenever the item NVT workflow is retrained.
"""

import pathlib
import logging

import numpy as np
import pandas as pd
import triton_python_backend_utils as pb_utils

logger = logging.getLogger("item_id_decoder")


class TritonPythonModel:
    def initialize(self, args):
        repository_path = pathlib.Path(args["model_repository"])
        if str(repository_path).endswith(".py"):
            repository_path = repository_path.parent.parent
        ver = args["model_version"]

        parquet_path = repository_path / ver / "categories" / "unique.item_id.parquet"
        mapping_df = pd.read_parquet(parquet_path)
        max_nvt_idx = mapping_df.index.max()
        self.decode_table = np.zeros(max_nvt_idx + 1, dtype=np.int32)
        for nvt_idx, row in mapping_df.iterrows():
            self.decode_table[nvt_idx] = row["item_id"]
        
        # pb_utils.Logger.log_info(
        #     f"item_id_decoder ready: max_nvt_idx={max_nvt_idx}, "
        #     f"sample: nvt[3]->{self.decode_table[3]}, "
        #     f"nvt[1802]->{self.decode_table[min(1802, max_nvt_idx)]}"
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
        nvt_ids = pb_utils.get_input_tensor_by_name(
            request, "ordered_ids"
        ).as_numpy().reshape(-1)

        # pb_utils.Logger.log_warn(f"NVT-encoded item ids received by ID decoder model: {nvt_ids.tolist()}")
        scores = pb_utils.get_input_tensor_by_name(
            request, "ordered_scores"
        ).as_numpy().reshape(-1)

        # Decode: clamp to valid range, then lookup
        clamped = np.clip(nvt_ids, 0, len(self.decode_table) - 1) #not really necessary as the NVTworkflow item vocab cannot be misaligned with the decoder table (unique item ids) since they are both updated together during training.
        original_ids = self.decode_table[clamped]
        # pb_utils.Logger.log_warn(f"Original item ids decoded by ID decoder model: {original_ids.tolist()}")

        return pb_utils.InferenceResponse([
            pb_utils.Tensor("ordered_ids", original_ids.astype(np.int32)),
            pb_utils.Tensor("ordered_scores", scores.astype(np.float32)),
        ])