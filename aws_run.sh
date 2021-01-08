aws ecs run-task \
  --cluster QTrader-Cluster \
  --task-definition QTraderECSScheduledTaskScheduledTaskDef052D47E0:5 \
  --launch-type "FARGATE" \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-0e5cd4294688fc6b3,subnet-070e75597e6466421],securityGroups=[sg-068ef2ad4a89c682d],assignPublicIp=DISABLED}"
