from kfp import dsl
from kfp import compiler
from kfp import kubernetes
import argparse
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def get_copy_incremental_component(image: str):
    @dsl.container_component
    def copy_incremental(
        local_data_path: str,
        aws_region: str,
        s3_bucket: str,
        new_data_s3_path: str,
        replay_data_s3_path: str,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["bash", "/script/copy_incremental.sh"],
            args=[
                local_data_path,
                aws_region,
                s3_bucket,
                new_data_s3_path,
                replay_data_s3_path,
            ]
        )
    return copy_incremental


def get_preprocess_incremental_component(image: str):
    @dsl.container_component
    def preprocess_incremental(
        local_data_path: str,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["bash", "/script/preprocess_incremental.sh"],
            args=[
                local_data_path,
            ]
        )
    return preprocess_incremental


def get_train_incremental_component(image: str):
    @dsl.container_component
    def train_incremental(
        local_data_path: str,
        aws_region: str,
        s3_bucket: str,
        num_epochs: int,
        learning_rate: float,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["bash", "/script/train_incremental.sh"],
            args=[
                local_data_path,
                aws_region,
                s3_bucket,
                num_epochs,
                learning_rate,
            ]
        )
    return train_incremental


def get_promote_incremental_component(image: str):
    @dsl.container_component
    def promote_incremental(
        triton_service_host: str,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["bash", "/script/promote_incremental.sh"],
            args=[
                triton_service_host,
            ]
        )
    return promote_incremental


def create_pipeline(data_copy_image: str, etl_train_image: str):
    copy_op = get_copy_incremental_component(data_copy_image)
    preprocess_op = get_preprocess_incremental_component(etl_train_image)
    train_op = get_train_incremental_component(etl_train_image)
    promote_op = get_promote_incremental_component(data_copy_image)

    @dsl.pipeline(
        name="Incremental Fine-Tuning Pipeline for the Multistage Recommender System",
        description="Fetches new interaction data, runs incremental preprocessing, fine-tunes the query tower and DLRM, and promotes updated models to Triton."
    )
    def incremental_recsys_pipeline(
        s3_bucket: str = "your-s3-bucket-name",
        local_data_path: str = "/var/lib/data",
        aws_region: str = "us-east-1",
        new_data_s3_path: str = "new_data/",
        replay_data_s3_path: str = "old_merged/",
        num_epochs: int = 1,
        learning_rate: float = 1e-4,
        triton_service_host: str = "triton-service.default.svc.cluster.local",
    ):
        persistent_volume_claim_name = "my-cluster-pvc"
        mount_path = "/var/lib/data"

        # Step 1:download new_data and old_merged from S3
        copy_task = copy_op(
            local_data_path=local_data_path,
            aws_region=aws_region,
            s3_bucket=s3_bucket,
            new_data_s3_path=new_data_s3_path,
            replay_data_s3_path=replay_data_s3_path,
        ).set_caching_options(False).set_env_variable(name="HOME", value="/tmp")
        copy_task.set_display_name("Copy Incremental Data")

        kubernetes.mount_pvc(copy_task, pvc_name=persistent_volume_claim_name, mount_path=mount_path)
        kubernetes.add_node_selector(copy_task, label_key="karpenter.sh/nodepool", label_value="cpu-node-pool")

        # Step 2:feature fetch + NVT transform
        preprocess_task = preprocess_op(
            local_data_path=local_data_path,
        ).set_caching_options(False).set_env_variable(name="HOME", value="/tmp")
        preprocess_task.set_display_name("Preprocess Data")

        kubernetes.mount_pvc(preprocess_task, pvc_name=persistent_volume_claim_name, mount_path=mount_path)
        kubernetes.add_node_selector(preprocess_task, label_key="karpenter.sh/nodepool", label_value="gpu-node-pool")
        preprocess_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
        kubernetes.add_toleration(preprocess_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")

        # Step 3:fine-tune query tower and DLRM, version and upload to S3
        train_task = train_op(
            local_data_path=local_data_path,
            aws_region=aws_region,
            s3_bucket=s3_bucket,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
        ).set_caching_options(False).set_env_variable(name="HOME", value="/tmp")
        train_task.set_display_name("Finetune Models")
        kubernetes.mount_pvc(train_task, pvc_name=persistent_volume_claim_name, mount_path=mount_path)
        kubernetes.add_node_selector(train_task, label_key="karpenter.sh/nodepool", label_value="gpu-node-pool")
        train_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
        kubernetes.add_toleration(train_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")

        # Step 4:promote new model versions to Triton via model control API
        promote_task = promote_op(
            triton_service_host=triton_service_host,
        ).set_caching_options(False).set_env_variable(name="HOME", value="/tmp")
        promote_task.set_display_name("Promote Models")

        kubernetes.mount_pvc(promote_task, pvc_name=persistent_volume_claim_name, mount_path=mount_path)
        kubernetes.add_node_selector(promote_task, label_key="karpenter.sh/nodepool", label_value="cpu-node-pool")

        for task in [copy_task, preprocess_task, train_task, promote_task]:
            kubernetes.set_image_pull_policy(task, "Always")

        preprocess_task.after(copy_task)
        train_task.after(preprocess_task)
        promote_task.after(train_task)

    return incremental_recsys_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dcoi",
                        "--data-copy-image",
                        type=str,
                        required=True,
                        help="ECR URI for the data copy image (used for copy and promote steps)")
    parser.add_argument("-eti",
                        "--etl-train-image",
                        type=str,
                        required=True,
                        help="ECR URI for the ETL train image (used for preprocess and train steps)")
    args = parser.parse_args()

    logger.info("Data copy image: %s", args.data_copy_image)
    logger.info("ETL train image: %s", args.etl_train_image)

    pipeline = create_pipeline(
        data_copy_image=args.data_copy_image,
        etl_train_image=args.etl_train_image,
    )

    output_file = "incremental_recsys_pipeline.yaml"
    compiler.Compiler().compile(pipeline, output_file)
    logger.info("Pipeline definition written to %s", output_file)
