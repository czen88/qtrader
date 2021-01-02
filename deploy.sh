# Generate Requirements
pipenv lock --requirements > requirements.txt

# Remove old images
docker image prune -f
docker image rm -f $AWS_ECR_URL

# Build Docker Image
docker build -t q-trader .

# Tag Docker Image
docker tag q-trader:latest $AWS_ECR_URL

# Login to ECR via Docker
$(aws ecr get-login --no-include-email --region $AWS_REGION)

# Push Docker Image to ECR
docker push $AWS_ECR_URL
