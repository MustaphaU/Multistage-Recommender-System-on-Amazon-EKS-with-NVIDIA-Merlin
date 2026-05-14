AWS_REGION=${1:-"us-east-1"}
AWS_ACCOUNT_ID=${2:-"ACCOUNT_ID"}

aws ecr create-repository --repository-name recsys-lambda --region $AWS_REGION || true
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker build --platform linux/amd64 --provenance=false -f Dockerfile.lambda -t recsys-lambda .
docker tag recsys-lambda:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/recsys-lambda:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/recsys-lambda:latest