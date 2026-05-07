"""
Model 5b: Item NVTabular Transform (Python Backend)
=====================================================
NVTabular item subworkflow: Categorify + LogNorm.

Input:  item_id, price, category_l1, category_l2, item_gender  (raw Feast features)
Output: item_id, category_l1, category_l2, item_gender, price_log_norm  (nvt transformed)

Artifacts:
    1/workflow/  — NVTabular item subworkflow

Update cadence: Retrained/exported on schema changes or new features.
"""

import json
import pathlib
import logging

import numpy as np
import triton_python_backend_utils as pb_utils

# logger = logging.getLogger("5b_item_nvt")


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
        # logger.info("5b_item_nvt ready  (workflow=%s)", workflow_path)

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
            "item_id":     pb_utils.get_input_tensor_by_name(request, "item_id").as_numpy().reshape(-1),
            "price":       pb_utils.get_input_tensor_by_name(request, "price").as_numpy().reshape(-1),
            "category_l1": pb_utils.get_input_tensor_by_name(request, "category_l1").as_numpy().reshape(-1),
            "category_l2": pb_utils.get_input_tensor_by_name(request, "category_l2").as_numpy().reshape(-1),
            "item_gender": pb_utils.get_input_tensor_by_name(request, "item_gender").as_numpy().reshape(-1),
        }
        #pb_utils.Logger.log_warn(f"item ids received by item NVT model: {input_tensors['item_id'].tolist()}")
        transformed = self.runner.run_workflow(input_tensors)
        output_tensors = [pb_utils.Tensor(name, data) for name, data in transformed.items()]
        return pb_utils.InferenceResponse(output_tensors)
