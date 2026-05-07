"""
Model 1c: Context NVTabular Transform (Python Backend)
=======================================================
NVTabular context subworkflow: Categorify device_type,
derive cyclical time features from raw timestamp.

Input:  device_type (TYPE_INT32, [-1])  : "mobile"/"tablet"/"desktop"/"unknown" (integer codes)
        timestamp   (TYPE_INT32, [-1])  — unix epoch seconds
Output: device_type     (TYPE_INT32, [-1]) — categorified
        hour_sin        (TYPE_FP32,  [-1])
        hour_cos        (TYPE_FP32,  [-1])
        dow_sin         (TYPE_FP32,  [-1])
        dow_cos         (TYPE_FP32,  [-1])

Artifacts:
    1/workflow/  — NVTabular context subworkflow (contains Categorify vocab) -> use the full workflow object, unless you saved the context subgraph separately, also ensure to include the categories parquet files

Update cadence: Retrained/exported on schema changes or new features emerge.
"""

import json
import pathlib
import logging

import numpy as np
import triton_python_backend_utils as pb_utils

logger = logging.getLogger("1c_context_nvt")


class TritonPythonModel:
    def initialize(self, args):
        import nvtabular
        from merlin.systems.workflow.base import WorkflowRunner

        repository_path = pathlib.Path(args["model_repository"])
        if str(repository_path).endswith(".py"):
            repository_path = repository_path.parent.parent
        ver = args["model_version"]
        model_device = args["model_instance_kind"]
        self.model_config = json.loads(args["model_config"])

        workflow_path = repository_path / ver / "workflow"
        self.workflow = nvtabular.Workflow.load(str(workflow_path))

        self.output_dtypes = {}
        for col_name in self.workflow.output_schema.column_names:
            conf = pb_utils.get_output_config_by_name(self.model_config, col_name)
            self.output_dtypes[col_name] = pb_utils.triton_string_to_numpy(conf["data_type"])

        self.runner = WorkflowRunner(
            self.workflow, self.output_dtypes, self.model_config, model_device,
        )
        # logger.info("1c_context_nvt ready  (workflow=%s)", workflow_path)

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
        input_tensors = {
            "device_type": pb_utils.get_input_tensor_by_name(request, "device_type").as_numpy().reshape(-1),
            "timestamp":   pb_utils.get_input_tensor_by_name(request, "timestamp").as_numpy().reshape(-1),
        }
        transformed = self.runner.run_workflow(input_tensors)
        output_tensors = [pb_utils.Tensor(name, data) for name, data in transformed.items()]
        return pb_utils.InferenceResponse(output_tensors)
