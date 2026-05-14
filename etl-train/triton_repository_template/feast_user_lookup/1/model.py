"""
Model 1a: Feast User Lookup (Python Backend)
==============================================
Feast online store → raw user features, with an optional live top-category
override derived from recent seen items stored in Redis.

Input:  user_id  (TYPE_INT32, [-1])
Output: user_id  (TYPE_INT32, [-1])
        age      (TYPE_INT32, [-1])
        gender   (TYPE_INT32, [-1])
    top_category (TYPE_INT32, [-1])

Environment:
    FEAST_REPO_PATH — Path to Feast feature repository

Model config parameters:
    DEFAULT_USER_AGE — Age value for unknown users (set by train.sh
                       from NVT train stats). Falls back to 36.

Graceful degradation:
    If a user_id is not found in Feast (cold-start), returns defaults:
    user_id=-1 (OOV sentinel), age=DEFAULT_USER_AGE, gender=-1 (OOV), top_category=-1 (OOV)
    The NVT workflow maps these to OOV embeddings that the model
    was trained to handle via 5% random OOV masking.

Update cadence: Feast online store refreshed by hourly materialization job.
"""

import os
import json
import logging

import numpy as np
import triton_python_backend_utils as pb_utils

#logger = logging.getLogger("1a_feast_user")

# OOV sentinel for categorical features (gender, user_id): matches our OOV mask value before nvt transform.
OOV_SENTINEL = -1


class TritonPythonModel:
    def initialize(self, args):
        import feast

        model_config = json.loads(args["model_config"])

        # Read default age from model config (injected by train.sh)
        raw_age=(
            model_config.get("parameters", {})
            .get("DEFAULT_USER_AGE", {})
            .get("string_value")
        )

        if raw_age is None:
            self.default_age = 36
            pb_utils.Logger.log_warn(
                f"DEFAULT_USER_AGE not found in model config parameters; using fallback default_age={self.default_age}"
            )
        else:
            try:
                self.default_age = int(raw_age)
            except ValueError:
                self.default_age = 36
                pb_utils.Logger.log_warn(
                    f"Invalid DEFAULT_USER_AGE value '{raw_age}' in model config parameters; using fallback default_age={self.default_age}"
                )
        feast_repo = "/model/script/feast_repo/feature_repo"
        self.store = feast.FeatureStore(feast_repo)
        self.feature_refs = ["user_features:age", "user_features:gender", "user_features:top_category"]
        # logger.info(
        #     "1a_feast_user ready  (feast=%s, default_age=%d)",
        #     feast_repo, self.default_age,
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
        user_ids = pb_utils.get_input_tensor_by_name(request, "user_id").as_numpy().reshape(-1)

        entity_rows = [{"user_id": int(uid)} for uid in user_ids]
        feast_dict = self.store.get_online_features(
            features=self.feature_refs,
            entity_rows=entity_rows,
        ).to_dict()

        # Fill nulls for unknown users with OOV defaults
        out_user_ids = []
        out_ages = []
        out_genders = []
        out_top_cats = []
        for i, uid in enumerate(feast_dict["user_id"]):
            age = feast_dict["age"][i]
            gender = feast_dict["gender"][i]
            top_cat = feast_dict["top_category"][i]
            if age is None or gender is None:
                # Cold-start user: use OOV defaults
                out_user_ids.append(OOV_SENTINEL)
                out_ages.append(self.default_age)
                out_genders.append(OOV_SENTINEL)
                out_top_cats.append(OOV_SENTINEL)
            else:
                out_user_ids.append(uid)
                out_ages.append(age)
                out_genders.append(gender)
                out_top_cats.append(top_cat if top_cat is not None else OOV_SENTINEL)

        return pb_utils.InferenceResponse([
            pb_utils.Tensor("user_id",      np.array(out_user_ids, dtype=np.int32)),
            pb_utils.Tensor("age",          np.array(out_ages,     dtype=np.int32)),
            pb_utils.Tensor("gender",       np.array(out_genders,  dtype=np.int32)),
            pb_utils.Tensor("top_category", np.array(out_top_cats, dtype=np.int32)),
        ])
