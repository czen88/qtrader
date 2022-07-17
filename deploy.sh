IMAGE=$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$(aws configure get region).amazonaws.com/qtrader/service:latest

# Remove old images.
# TODO: clean properly
docker image prune -f

# Build Docker Image
docker build -t $IMAGE .

#Test Docker Locally. Do not forget to delete old images first!
#docker-compose up

# Login to ECR with Docker
aws ecr get-login-password --region $AWS_REGION \
| docker login \
    --username AWS \
    --password-stdin 266976398848.dkr.ecr.$AWS_REGION.amazonaws.com

# Push Docker Image to ECR
docker push $IMAGE
