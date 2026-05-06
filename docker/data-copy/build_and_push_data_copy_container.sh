set -e

AWS_ACCOUNT_ID=$1
AWS_REGION=${2:-"us-east-1"}
IMAGE_TAG="0.0.0"

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
REPO_NAME="merlin-multistage-recsys/data-copy"
FULL_IMAGE_NAME="${ECR_REGISTRY}/${REPO_NAME}:${IMAGE_TAG}"

echo "Building data extraction component..."
echo "FULL IMAGE NAME: $FULL_IMAGE_NAME"

aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

aws ecr describe-repositories --repository-names $REPO_NAME --region $AWS_REGION >/dev/null 2>&1 || \
aws ecr create-repository --repository-name $REPO_NAME --region $AWS_REGION

cd "$(dirname "$0")/../.."

docker build --no-cache -f docker/data-copy/Dockerfile.copy -t $FULL_IMAGE_NAME .

docker push $FULL_IMAGE_NAME

mkdir -p .image_uris
echo $FULL_IMAGE_NAME > .image_uris/data_copy_image_uri.txt
echo "Image URI saved to .image_uris/data_copy_image_uri.txt"