"""
Model 4: Seen-Item Filter (Python Backend) — Bloom Filter
============================================
Pipeline stage 4 of 6.

Uses Redis/Valkey Bloom Filter (BF.MEXISTS) to check whether candidates
have been seen by the user. Much cheaper than ZSET at scale — O(1) per
lookup, tiny memory footprint, and false positives (occasionally filtering
an unseen item) are harmless in a rec system.

Input:  user_id        (TYPE_INT32, [-1])   — single user ID
        candidate_ids  (TYPE_INT32, [-1])   — FAISS candidates (e.g. 300)
Output: filtered_ids   (TYPE_INT32, [-1])   — candidates minus seen items

External dependency:
    Valkey 8.2+ — Bloom filter key pattern: bf:seen:{user_id}

Environment:
    REDIS_URL      — Redis/Valkey connection string (default: "" → no filtering)
    BF_CAPACITY    — Expected number of items per user bloom filter (default: 1000)
    BF_ERROR_RATE  — False positive rate (default: 0.01)

Graceful degradation:
    If Redis/Valkey is unavailable or REDIS_URL is empty, ALL candidates
    pass through (filter is a no-op).
"""

import os
import logging

import numpy as np
import triton_python_backend_utils as pb_utils

logger = logging.getLogger("4_seen_filter")


class TritonPythonModel:
    def initialize(self, args):
        """Connect to Valkey (optional)."""
        import json
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        self.redis_url = (
            params.get("REDIS_URL", {}).get("string_value")
        )
        self.bf_capacity = 1000
        self.bf_error_rate = 0.01
        self._client = None

        if self.redis_url:
            try:
                import redis

                self._client = redis.Redis.from_url(
                    self.redis_url, decode_responses=True
                )
                self._client.ping()
                pb_utils.Logger.log_info(
                    f"4_seen_filter ready (valkey={self.redis_url}, bloom mode, capacity={self.bf_capacity}, error_rate={self.bf_error_rate:.3f})"
                )
            except Exception as exc:
                pb_utils.Logger.log_warn(
                    f"4_seen_filter: Valkey unavailable ({exc}), filter disabled"
                )
                self._client = None
        else:
            pb_utils.Logger.log_warn("4_seen_filter ready (no Valkey — passthrough mode)")

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
        user_ids = (
            pb_utils.get_input_tensor_by_name(request, "user_id")
            .as_numpy()
            .reshape(-1)
        )
        candidate_ids = (
            pb_utils.get_input_tensor_by_name(request, "candidate_ids")
            .as_numpy()
            .reshape(-1)
        )

        uid = int(user_ids[0])
        seen_mask = self._check_seen_bloom(uid, candidate_ids)

        if seen_mask is not None:
            filtered = candidate_ids[~seen_mask]
        else:
            filtered = candidate_ids

        return pb_utils.InferenceResponse(
            [pb_utils.Tensor("filtered_ids", filtered.astype(np.int32))]
        )

    def _check_seen_bloom(self, user_id: int, candidate_ids: np.ndarray):
        """
        Batch-check candidates against the user's Bloom filter using BF.MEXISTS.
        Returns a boolean numpy array (True = seen) or None if unavailable.
        """
        if self._client is None or len(candidate_ids) == 0:
            return None
        try:
            key = f"bf:seen:{user_id}"
            results = self._client.execute_command(
                "BF.MEXISTS", key, *candidate_ids.tolist()
            )
            return np.array(results, dtype=bool)
        except Exception as exc:
            pb_utils.Logger.log_warn(f"Bloom check failed for user {user_id}: {exc}")
            self._client = None
            return None
        