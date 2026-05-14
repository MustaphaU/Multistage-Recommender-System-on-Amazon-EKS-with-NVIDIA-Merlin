AWS_REGION=${1:-"us-east-1"}
AWS_ACCOUNT_ID=${2:-"ACCOUNT_ID"}

aws ecr create-repository --repository-name recsys-feature-computation --region $AWS_REGION || true
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker build --platform linux/amd64 --provenance=false -f Dockerfile.feature_computation -t recsys-feature-computation .
docker tag recsys-feature-computation:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/recsys-feature-computation:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/recsys-feature-computation:latest
