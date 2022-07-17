aws ecs run-task \
  --cluster QTrader-Cluster \
  --task-definition QTraderECSScheduledTaskScheduledTaskDef052D47E0:8 \
  --launch-type "FARGATE" \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-0f1d916c00a718735],securityGroups=[sg-0550b594d0a4d8513],assignPublicIp=ENABLED}"
