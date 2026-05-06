"""
Model 0: Context Preprocessor (Python Backend)
================================================
Fills default values for optional context inputs.

If device_type or timestamp are missing or contain sentinel values,
this step substitutes OOV defaults that match what the model saw
during training (refer to the 5% context masking in preprocess.py).

Input:  device_type (TYPE_INT32, [-1])  — an integer code for the device type
        timestamp   (TYPE_INT32, [-1])  — unix epoch, Integer
Output: device_type (TYPE_INT32, [-1])  — cleaned
        timestamp   (TYPE_INT32, [-1])  — cleaned

This is a lightweight passthrough that ensures downstream NVT
context transform always gets valid inputs.
"""

import logging
import time

import numpy as np
import triton_python_backend_utils as pb_utils

logger = logging.getLogger("0_context_preproc")


class TritonPythonModel:
    def initialize(self, args):
        logger.info("0_context_preproc ready")

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
        # Device type: default to -1 (OOV sentinel) if missing
        device_tensor = pb_utils.get_input_tensor_by_name(request, "device_type")
        if device_tensor is not None:
            device = device_tensor.as_numpy().reshape(-1).astype(np.int32)
        else:
            device = np.array([-1], dtype=np.int32)

        # Timestamp: default to current server time if missing or 0
        ts_tensor = pb_utils.get_input_tensor_by_name(request, "timestamp")
        if ts_tensor is not None:
            ts = ts_tensor.as_numpy().reshape(-1).astype(np.int32)
            ts = np.where(ts == 0, int(time.time()), ts)
        else:
            ts = np.array([int(time.time())], dtype=np.int32)

        return pb_utils.InferenceResponse([
            pb_utils.Tensor("device_type", device.astype(np.int32)),
            pb_utils.Tensor("timestamp",   ts.astype(np.int32)),
        ])
