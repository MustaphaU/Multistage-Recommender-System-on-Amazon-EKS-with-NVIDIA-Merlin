from kfp import dsl
from kfp import compiler
from kfp import kubernetes
import argparse
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def get_data_copy_component(image: str):
    @dsl.container_component
    def data_copy(
        s3_bucket: str,
        local_data_path: str,
        aws_region: str,
        s3_initial_path: str,
        s3_incremental_path: str,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["bash", "/script/copy_to_pvc.sh"],
            args=[
                s3_bucket,
                local_data_path,
                aws_region,
                s3_initial_path,
                s3_incremental_path,
            ]
        )

    return data_copy

def get_preprocess_component(image: str):
    @dsl.container_component
    def preprocess(
        local_data_path: str,
        aws_region: str,
        s3_bucket: str,
        glue_database: str,
        athena_workgroup: str,
        s3_incremental_path: str,
        redis_host: str,
        redis_port: int,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["bash", "/script/preprocess.sh"],
            args=[
                local_data_path,
                aws_region,
                s3_bucket,
                glue_database,
                athena_workgroup,
                s3_incremental_path,
                redis_host,
                redis_port,
            ]
        )

    return preprocess


def get_training_component(image: str):
    @dsl.container_component
    def train(
        local_data_path: str,
        aws_region: str,
        s3_bucket: str,
        redis_host: str,
        redis_port: int,
        retrieval_topk: int,
        ranking_topk: int,
        diversity_mode: str,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["bash", "/script/train.sh"],
            args=[
                local_data_path,
                aws_region,
                s3_bucket,
                redis_host,
                redis_port,
                retrieval_topk,
                ranking_topk,
                diversity_mode,
            ]
        )

    return train


def get_deployment_component(image: str):
    @dsl.container_component
    def deploy(
        local_data_path: str,
        aws_region: str,
        triton_inference_image_in_ecr: str,
        service_account: str,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["bash", "/script/triton-inference/deploy_triton_with_helm.sh"],
            args=[
                local_data_path,
                aws_region,
                triton_inference_image_in_ecr,
                service_account,
            ]
        )

    return deploy

def create_pipeline(data_copy_image: str, preprocess_image: str, train_image: str, deploy_image: str):
    data_copy_op = get_data_copy_component(data_copy_image)
    preprocess_op = get_preprocess_component(preprocess_image)
    train_op = get_training_component(train_image)
    deploy_op = get_deployment_component(deploy_image)

    @dsl.pipeline(
        name="Kubeflow Pipeline for the Multistage Recommender System",
        description="The Pipeline orchestrates the full ETL, training, and deployment workflow for the Multistage Recommender System on EKS."
    )
    def multistage_recsys_pipeline(
        s3_bucket: str = 'your-s3-bucket-name-for-training-data-and-features',
        local_data_path: str = 'PVC mount path e.g., /var/lib/data',
        aws_region: str = 'e.g., us-east-1',
        s3_initial_path: str = 'path-in-s3-bucket-for-initial-training-data',
        s3_incremental_path: str = 'path-in-s3-bucket-for-incremental-data',
        glue_database: str = 'your-glue-database-name-for-feast',
        athena_workgroup: str = 'your-athena-workgroup-name-for-feast',
        redis_host: str = 'your-redis-host-for-valkey e.g., master.xxxx-xxxx.xxxx.use1.cache.amazonaws.com',
        redis_port: int = 6379,
        retrieval_topk: int = 300,
        ranking_topk: int = 100,
        diversity_mode: str = 'false',
        triton_inference_image_in_ecr: str = 'your-ecr-repo-uri-for-triton-inference-image',
        service_account: str = 'merlin-kfp-sa'
    ):
        persistent_volume_claim_name = "my-cluster-pvc"  # ensure this PVC exists in your cluster
        mount_path = "/var/lib/data"

        # First task: Copy data from S3 to PVC (local storage in cluster)
        data_copy_task = data_copy_op(
            s3_bucket=s3_bucket,
            local_data_path=local_data_path,
            aws_region=aws_region,
            s3_initial_path=s3_initial_path,
            s3_incremental_path=s3_incremental_path,
        ).set_caching_options(False).set_env_variable(name='HOME', value='/tmp')

        # add volume, node selector for data copy task
        kubernetes.mount_pvc(
            data_copy_task,
            pvc_name=persistent_volume_claim_name,
            mount_path=mount_path
        )
        kubernetes.add_node_selector(
            data_copy_task,
            label_key="karpenter.sh/nodepool",
            label_value="cpu-node-pool"
        )

        # Second task: Preprocess data
        preprocess_task = preprocess_op(
            local_data_path=local_data_path,
            aws_region=aws_region,
            s3_bucket=s3_bucket,
            s3_incremental_path=s3_incremental_path,
            redis_host=redis_host,
            redis_port=redis_port,
            glue_database=glue_database,
            athena_workgroup=athena_workgroup,
        ).set_caching_options(False).set_env_variable(name='HOME', value='/tmp')

        # add volume, node selector for preprocess task
        kubernetes.mount_pvc(
            preprocess_task,
            pvc_name=persistent_volume_claim_name,
            mount_path=mount_path
        )
        kubernetes.add_node_selector(
            preprocess_task,
            label_key="karpenter.sh/nodepool",
            label_value="gpu-node-pool"
        )
        #select accelerator and add toleration for GPU node taint
        preprocess_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
        kubernetes.add_toleration(
            preprocess_task,
            key="nvidia.com/gpu",
            operator="Exists",
            effect="NoSchedule"
        )

        # Third task: Train model
        train_task = train_op(
            local_data_path=local_data_path,
            aws_region=aws_region,
            s3_bucket=s3_bucket,
            redis_host=redis_host,
            redis_port=redis_port,
            retrieval_topk=retrieval_topk,
            ranking_topk=ranking_topk,
            diversity_mode=diversity_mode,
        ).set_caching_options(False).set_env_variable(name='HOME', value='/tmp')
        # add volume, node selector for training task
        kubernetes.mount_pvc(
            train_task,
            pvc_name=persistent_volume_claim_name,
            mount_path=mount_path
        )
        kubernetes.add_node_selector(
            train_task,
            label_key="karpenter.sh/nodepool",
            label_value="gpu-node-pool"
        )
        #select accelerator and add toleration for GPU node taint
        train_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
        kubernetes.add_toleration(
            train_task,
            key="nvidia.com/gpu",
            operator="Exists",
            effect="NoSchedule"
        )

        # Fourth task: Deploy triton
        deploy_task = deploy_op(
            local_data_path=local_data_path,
            aws_region=aws_region,
            triton_inference_image_in_ecr=triton_inference_image_in_ecr,
            service_account=service_account,
        ).set_caching_options(False).set_env_variable(name='HOME', value='/tmp')
        # add volume, node selector for deployment task
        kubernetes.mount_pvc(
            deploy_task,
            pvc_name=persistent_volume_claim_name,
            mount_path=mount_path
        )
        kubernetes.add_node_selector(
            deploy_task,
            label_key="karpenter.sh/nodepool",
            label_value="cpu-node-pool"
        )


        #pull policy for all tasks - always
        for task in [data_copy_task, preprocess_task, train_task, deploy_task]:
            kubernetes.set_image_pull_policy(task, "Always")

        #define dependencies between tasks
        preprocess_task.after(data_copy_task)
        train_task.after(preprocess_task)
        deploy_task.after(train_task)

    return multistage_recsys_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dcoi", "--data-copy-image",
        type=str,
        required=True,
        help="ECR URI for the data copy task image"
    )
    parser.add_argument(
        "-ppi", "--preprocess-image",
        type=str,
        required=True,
        help="ECR URI for the preprocess task image"
    )
    parser.add_argument(
        "-ti", "--train-image",
        type=str,
        required=True,
        help="ECR URI for the training task image, same as preprocess image"
    )
    parser.add_argument(
        "-di", "--deploy-image",
        type=str,
        required=True,
        help="ECR URI for the deployment task image, use same container as data copy"
    )
    args = parser.parse_args()

    #log image urls
    logger.info(f"Data copy image: {args.data_copy_image}")
    logger.info(f"Preprocess image: {args.preprocess_image}")
    logger.info(f"Train image: {args.train_image}")
    logger.info(f"Deploy image: {args.deploy_image}")

    pipeline = create_pipeline(
        data_copy_image=args.data_copy_image,
        preprocess_image=args.preprocess_image,
        train_image=args.train_image,
        deploy_image=args.deploy_image
    )

    output_file = "multistage_recsys_pipeline.yaml"
    compiler.Compiler().compile(pipeline, output_file)
    logger.info(f"Pipeline definition has been written to {output_file}")