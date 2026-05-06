"""
Model 3: Retrieval (Python Backend)
====================================
Pipeline stage 3 of 6.

FAISS approximate nearest-neighbor search.

Input:  output_1       (TYPE_FP32, [-1, 64])  — user embedding from query tower
Output: candidate_ids  (TYPE_INT32, [-1])     — top-K item IDs from the index

Artifacts:
    1/index.faiss  — FAISS IVF32,Flat index with item_id mapping

Update cadence:
    - FAISS index: rebuilt weekly when the item tower is retrained
"""

import pathlib
import logging

import numpy as np
import triton_python_backend_utils as pb_utils

# logger = logging.getLogger("3_retrieval")

class TritonPythonModel:
    def initialize(self, args):
        """Load FAISS index."""
        import json
        import faiss

        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        self.topk = int(params.get("RETRIEVAL_TOPK", {}).get("string_value", "50"))

        repository_path = pathlib.Path(args["model_repository"])
        if str(repository_path).endswith(".py"):
            repository_path = repository_path.parent.parent
        ver = args["model_version"]

        index_path = str(repository_path / ver / "index.faiss")
        self.index = faiss.read_index(index_path)

        # logger.info(
        #     "3_retrieval ready  (d=%d, ntotal=%d, topk=%d)",
        #     self.index.d, self.index.ntotal, self.topk,
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
        # User embedding from the query tower  (shape: [1, 64])
        user_vector = pb_utils.get_input_tensor_by_name(
            request, "output_1"
        ).as_numpy().reshape(1, -1).astype(np.float32)

        # FAISS search — returns (distances, item_ids)
        # The index was built with add_with_ids so indices ARE item_ids
        _, indices = self.index.search(user_vector, self.topk)
        candidate_ids = indices.astype(np.int32).reshape(-1)

        return pb_utils.InferenceResponse(
            [pb_utils.Tensor("candidate_ids", candidate_ids)]
        )
