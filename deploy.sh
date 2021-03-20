# Generate Requirements
pipenv lock --requirements > requirements.txt

IMAGE=$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$(aws configure get region).amazonaws.com/qtrader/service:latest

# Remove old images.
# TODO: clean properly
docker image prune -f

# Build Docker Image
docker build -t $IMAGE .

#Test Docker Locally. Do not forget to delete old images first!
#docker-compose up

# Login to ECR via Docker
# See https://aws.amazon.com/blogs/compute/a-guide-to-locally-testing-containers-with-amazon-ecs-local-endpoints-and-docker-compose/
$(aws ecr get-login --no-include-email --region $AWS_REGION)

# Push Docker Image to ECR
docker push $IMAGE
