PV_LOC=${1:-"/var/lib/data"}
AWS_REGION=${2:-"us-east-1"}
TRITON_INFERENCE_IMAGE_URI=${3:-"123456789012.dkr.ecr.us-east-1.amazonaws.com/merlin-multistage-inference:0.0.0"}
SERVICE_ACCOUNT=${4:-"injected_by_deploy_component"}

if ! [ -d $PV_LOC/inference ]; then
    mkdir -p $PV_LOC/inference
    echo "Created inference directory at $PV_LOC/inference"
fi

triton_status=$(helm status triton 2>&1)
echo "Triton status check: "
echo "$triton_status"

if [[ "$triton_status" != *"STATUS: deployed"* ]]; then
    echo "Triton is not running. Deploying new instance..."
    
    #copied so that when the pod completes (is orphaned), the triton spec (in deployment.yaml) can still find the start_triton.sh script to run the triton server
    cp /script/triton-inference/triton-helm/start_triton.sh $PV_LOC/inference/start_triton.sh


    echo "Deploying Triton with image: $TRITON_INFERENCE_IMAGE_URI"
    
    helm install triton /script/triton-inference/triton-helm/ \
        --set image.imageName=$TRITON_INFERENCE_IMAGE_URI \
        --set service_account=$SERVICE_ACCOUNT
else
    echo "Triton is already running, not deploying another instance."
fi
