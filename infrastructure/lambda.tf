## DATA INGESTION: RDS SEEDER
resource "aws_cloudwatch_log_group" "rds_seeder" {
  name              = "/aws/lambda/${var.environment}-rds-seeder"
  retention_in_days = 14
}

resource "aws_lambda_function" "rds_seeder" {
  filename         = data.archive_file.rds_seeder_zip.output_path
  source_code_hash = data.archive_file.rds_seeder_zip.output_base64sha256
  function_name    = "${var.environment}-rds-seeder"
  role             = aws_iam_role.rds_seeder_role.arn
  handler          = "rds_seeder.lambda_handler"
  runtime          = "python3.12"
  memory_size      = 512
  timeout          = 300

  vpc_config {
    subnet_ids         = data.aws_subnets.private.ids
    security_group_ids = [aws_security_group.rds_seeder_sg.id]
  }

  environment {
    variables = {
      DB_CONFIG     = jsonencode(local.seed_config)
      DB_HOST       = aws_db_instance.dept_contacts_metadata.address
      DB_NAME       = aws_db_instance.dept_contacts_metadata.db_name
      DB_USER       = aws_db_instance.dept_contacts_metadata.username
      DB_SECRET_ARN = data.aws_secretsmanager_secret_version.dept_contacts_db_password.arn
      BUCKET_NAME   = aws_s3_bucket.dataset_storage.id
    }
  }

  depends_on = [aws_cloudwatch_log_group.rds_seeder]
}


## ORCHESTRATION LAYER
# This function manages the LangGraph state machine and coordinates tool calls.
resource "aws_cloudwatch_log_group" "orchestrator" {
  name              = "/aws/lambda/${var.environment}-orchestrator"
  retention_in_days = 14
}

resource "aws_lambda_function" "orchestrator" {
  filename         = data.archive_file.orchestrator_zip.output_path
  source_code_hash = data.archive_file.orchestrator_zip.output_base64sha256
  function_name    = "${var.environment}-orchestrator"
  role             = aws_iam_role.inference.arn
  handler          = "orchestrator.lambda_handler"
  runtime          = "python3.12"
  memory_size      = 1024
  timeout          = 120

  vpc_config {
    subnet_ids         = data.aws_subnets.private.ids
    security_group_ids = [aws_security_group.orchestrator.id]
  }

  environment {
    variables = {

      # URL for AgentCore
      AGENT_RUNTIME_ENDPOINT_URL = aws_vpc_endpoint.bedrock_agentcore.dns_entry[0]["dns_name"]
      # URL for Claude (Inference)
      BEDROCK_RUNTIME_ENDPOINT = aws_vpc_endpoint.bedrock.dns_entry[0]["dns_name"]
      # URL for Secrets Manager
      SECRETS_ENDPOINT_URL = aws_vpc_endpoint.secrets.dns_entry[0]["dns_name"]
      # Specific DNS for the Gateway Endpoint
      GATEWAY_ENDPOINT_URL = aws_vpc_endpoint.bedrock_gateway.dns_entry[0]["dns_name"]
      GATEWAY_URL          = "https://${aws_bedrockagentcore_gateway.tool_interface.gateway_id}.gateway.bedrock-agentcore.${var.aws_region}.amazonaws.com/mcp"
      MEMORY_ID            = aws_bedrockagentcore_memory.agent_chat_context.id
    }
  }

  depends_on = [aws_cloudwatch_log_group.orchestrator]
}


## TOOL LAYER: RDS SEARCH TOOL (MCP SERVER)
# Standardised interface for the Orchestrator to query the registry via Gateway.
resource "aws_cloudwatch_log_group" "rds_tool" {
  name              = "/aws/lambda/${var.environment}-rds-tool"
  retention_in_days = 14
}

resource "aws_lambda_function" "rds_tool" {
  filename         = data.archive_file.rds_tool_zip.output_path
  source_code_hash = data.archive_file.rds_tool_zip.output_base64sha256
  function_name    = "${var.environment}-rds-tool"
  role             = aws_iam_role.rds_tool_role.arn # Requires access to Bedrock and RDS
  handler          = "rds_tool.lambda_handler"
  runtime          = "python3.12"
  memory_size      = 512
  timeout          = 120

  vpc_config {
    subnet_ids         = data.aws_subnets.private.ids
    security_group_ids = [aws_security_group.rds_seeder_sg.id] # Reuse seeder SG for DB access
  }

  environment {
    variables = {
      DB_HOST       = aws_db_instance.dept_contacts_metadata.address
      DB_NAME       = aws_db_instance.dept_contacts_metadata.db_name
      DB_USER       = aws_db_instance.dept_contacts_metadata.username
      DB_SECRET_ARN = data.aws_secretsmanager_secret_version.dept_contacts_db_password.arn
    }
  }

  depends_on = [aws_cloudwatch_log_group.rds_tool]
}


## ORCHESTRATOR STREAMING ENDPOINT
# Enables the RESPONSE_STREAM mode for real-time interaction with the Svelte frontend.
resource "aws_lambda_function_url" "orchestrator_url" {
  function_name      = aws_lambda_function.orchestrator.function_name
  authorization_type = "NONE" # TODO: Recommend using AWS_IAM for production environments
  invoke_mode        = "RESPONSE_STREAM"

  cors {
    allow_credentials = true
    allow_origins     = ["*"] # TODO: Restrict to your specific frontend domain in production
    allow_methods     = ["POST"]
    allow_headers     = ["content-type"]
  }
}

output "orchestrator_url" {
  description = "The streaming HTTP endpoint for the Orchestrator"
  value       = aws_lambda_function_url.orchestrator_url.function_url
}
