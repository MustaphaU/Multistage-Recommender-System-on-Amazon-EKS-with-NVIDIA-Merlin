"""
Model 1b: User NVTabular Transform (Python Backend)
=====================================================
NVTabular user subworkflow: Normalize, Bucketize, Categorify.

Input:  user_id (TYPE_INT32), age (TYPE_INT32), gender (TYPE_INT32)
Output: user_id (TYPE_INT32), age_norm (TYPE_FP32), age_binned (TYPE_INT32), gender (TYPE_INT32)

Artifacts:
    1/workflow/  — NVTabular user subworkflow

Update cadence: Retrained/exported on schema changes or new features.
"""

import json
import pathlib
import logging

import numpy as np
import triton_python_backend_utils as pb_utils

# logger = logging.getLogger("1b_user_nvt")


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
        # logger.info("1b_user_nvt ready  (workflow=%s)", workflow_path)

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
            "user_id": pb_utils.get_input_tensor_by_name(request, "user_id").as_numpy().reshape(-1),
            "age":     pb_utils.get_input_tensor_by_name(request, "age").as_numpy().reshape(-1),
            "gender":  pb_utils.get_input_tensor_by_name(request, "gender").as_numpy().reshape(-1),
        }
        transformed = self.runner.run_workflow(input_tensors)
        output_tensors = [pb_utils.Tensor(name, data) for name, data in transformed.items()]
        return pb_utils.InferenceResponse(output_tensors)