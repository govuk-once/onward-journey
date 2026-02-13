## ORCHESTRATOR INFERENCE ROLE
# Primary execution role for the Onward Journey Orchestration Layer.

resource "aws_iam_role" "inference" {
  name               = "${var.environment}-inference-role"
  assume_role_policy = data.aws_iam_policy_document.allow_all_assume_role.json
}

data "aws_iam_policy_document" "allow_all_assume_role" {
  statement {
    sid = "AllowLambdaAndUsersToAssumeRole"

    actions = [
      "sts:AssumeRole"
    ]

    # This allows the Lambda Service to assume the role
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }

    # This allows local devs to assume the role for testing
    principals {
      type = "AWS"
      identifiers = [
        "arn:aws:iam::${var.aws_account_id}:root"
      ]
    }
  }
}

## INFERENCE ROLE ATTACHMENTS
# AWS managed policies and specific AgentCore access requirements.

# Grants the Orchestration Layer permission to create Network Interfaces within the VPC.
resource "aws_iam_role_policy_attachment" "lambda_vpc_access" {
  role       = aws_iam_role.inference.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

# Use AWS provided "Bedrock Limited Access" policy.
resource "aws_iam_role_policy_attachment" "inference_bedrock_access" {
  role       = aws_iam_role.inference.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonBedrockLimitedAccess"
}

# Permissions for the agent to interact with Amazon Bedrock modular capabilities (AgentCore).
resource "aws_iam_policy" "agentcore_access" {
  name        = "${var.environment}-agentcore-access"
  description = "Allows the Orchestration Layer to use managed Memory and Gateway modules."

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "MemoryAndGatewayOps"
        Effect = "Allow"
        Action = [
          # --- AGENTCORE MEMORY: SESSION & CONTEXT ---
          # Essential for LangGraph to save and retrieve conversation turns.
          "bedrock-agentcore:CreateSession",
          "bedrock-agentcore:GetSession",
          "bedrock-agentcore:CreateEvent",      # Action to save a turn to short-term memory
          "bedrock-agentcore:GetMemory",        # Action to retrieve tactical/strategic memory records
          "bedrock-agentcore:ListEvents",       # Required to list the checkpoint history
          "bedrock-agentcore:RetrieveMemories", # Required for more advanced context retrieval

          # --- AGENTCORE RUNTIME ---
          "bedrock:InvokeAgent",
          "bedrock-agentcore:InvokeAgentRuntime",
          "bedrock-agentcore:InvokeGateway", # Required for MCP POST calls
        ]
        Resource = [
          aws_bedrockagentcore_memory.agent_chat_context.arn,
          # The Gateway itself (for management)
          aws_bedrockagentcore_gateway.tool_interface.gateway_arn,
          # The Gateway sub-resources (for /runtime-endpoint/DEFAULT)
          "${aws_bedrockagentcore_gateway.tool_interface.gateway_arn}/*",
          # Identity boundary for agent execution
          "arn:aws:bedrock:${var.aws_region}:${var.aws_account_id}:agent-alias/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "inference_agentcore" {
  role       = aws_iam_role.inference.name
  policy_arn = aws_iam_policy.agentcore_access.arn
}

## ORCHESTRATOR RDS & SECRETS ACCESS
# Allows the Orchestrator to fetch the DB password and connect to the RDS instance.

resource "aws_iam_policy" "orchestrator_rds_secrets" {
  name        = "${var.environment}-orchestrator-rds-secrets"
  description = "Allows the Orchestrator to access RDS credentials and Bedrock streaming."

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "BedrockStreamingAccess"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = [
          # Permission for the base models
          "arn:aws:bedrock:${var.aws_region}::foundation-model/anthropic.claude-*",
          # Permission for the Inference Profiles
          "arn:aws:bedrock:${var.aws_region}:${var.aws_account_id}:inference-profile/eu.anthropic.claude-*",
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "inference_rds_secrets" {
  role       = aws_iam_role.inference.name
  policy_arn = aws_iam_policy.orchestrator_rds_secrets.arn
}

## DATASET ACCESS
# Permissions for reading source data from S3.

resource "aws_iam_policy" "dataset_read" {
  name        = "${var.environment}-dataset-read"
  description = "Allow read access to the dataset s3 bucket"
  policy      = data.aws_iam_policy_document.dataset_read.json
}

data "aws_iam_policy_document" "dataset_read" {
  statement {
    sid     = "ListBucket"
    actions = ["s3:ListBucket"]

    resources = [aws_s3_bucket.dataset_storage.arn]
  }

  statement {
    sid     = "ReadBucketObjects"
    actions = ["s3:GetObject"]

    resources = ["${aws_s3_bucket.dataset_storage.arn}/*"]
  }
}

resource "aws_iam_role_policy_attachment" "inference_allow_dataset_read" {
  role       = aws_iam_role.inference.name
  policy_arn = aws_iam_policy.dataset_read.arn
}


## BEDROCK AGENTCORE SERVICE ROLE
# Required for managed session memory and tool gateway connectivity.

resource "aws_iam_role" "agentcore_role" {
  name = "${var.environment}-agentcore-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = [
          "bedrock-agentcore.amazonaws.com",
          "bedrock.amazonaws.com"
        ]
      }
    }]
  })

  tags = {
    Name = "${var.environment}-agentcore-service-role"
  }
}

# Authorizes the Gateway to trigger the RDS Tool Lambda
resource "aws_iam_role_policy" "agentcore_gateway_invocation" {
  name = "${var.environment}-agentcore-gateway-invocation"
  role = aws_iam_role.agentcore_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowGatewayToInvokeTools"
        Effect = "Allow"
        Action = ["lambda:InvokeFunction"]
        Resource = [
          aws_lambda_function.rds_tool.arn,
          aws_lambda_function.genesys_tool.arn
        ]
      }
    ]
  })
}


## BEDROCK AGENTCORE RESOURCE-BASED POLICY
# Explicitly authorises the Bedrock service to invoke the RDS Tool Lambda.
# This acts as the "Resource-Based Policy" on the Lambda side.

resource "aws_lambda_permission" "allow_bedrock_gateway" {
  statement_id  = "AllowBedrockGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.rds_tool.function_name
  principal     = "bedrock-agentcore.amazonaws.com"
  source_arn    = aws_bedrockagentcore_gateway.tool_interface.gateway_arn
}

## BEDROCK AGENTCORE RESOURCE-BASED POLICY (GENESYS)
resource "aws_lambda_permission" "allow_bedrock_gateway_genesys" {
  statement_id  = "AllowBedrockGatewayInvokeGenesys"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.genesys_tool.function_name
  principal     = "bedrock-agentcore.amazonaws.com"
  source_arn    = aws_bedrockagentcore_gateway.tool_interface.gateway_arn
}

## RDS TOOL SERVICE ROLE
# Execution role for the MCP Tool Lambda that handles database searches.
resource "aws_iam_role" "rds_tool_role" {
  name               = "${var.environment}-rds-tool-role"
  assume_role_policy = data.aws_iam_policy_document.allow_all_assume_role.json
}

# Attachment: Reuse VPC access for private RDS connectivity
resource "aws_iam_role_policy_attachment" "rds_tool_vpc_access" {
  role       = aws_iam_role.rds_tool_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

# Attachment: Reuse existing RDS/Secrets/Embeddings policy
resource "aws_iam_role_policy_attachment" "rds_tool_main" {
  role       = aws_iam_role.rds_tool_role.name
  policy_arn = aws_iam_policy.rds_seeder_permissions.arn
}

## GENESYS TOOL SERVICE ROLE
# Execution role for the MCP Tool Lambda that handles Genesys API interactions.

resource "aws_iam_role" "genesys_tool_role" {
  name               = "${var.environment}-genesys-tool-role"
  assume_role_policy = data.aws_iam_policy_document.allow_all_assume_role.json
}

resource "aws_iam_policy" "genesys_tool_permissions" {
  name        = "${var.environment}-genesys-tool-permissions"
  description = "Allows the Genesys Tool to fetch OAuth credentials and log to CloudWatch."

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogAccess"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Sid      = "GenesysSecretAccess"
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = [aws_secretsmanager_secret.genesys_credentials.arn]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "genesys_tool_main" {
  role       = aws_iam_role.genesys_tool_role.name
  policy_arn = aws_iam_policy.genesys_tool_permissions.arn
}

## RDS SEEDER SERVICE ROLE
# Execution role for the Lambda responsible for database initialisation and data loading.

resource "aws_iam_role" "rds_seeder_role" {
  name = "${var.environment}-rds-seeder-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

## RDS SEEDER DATA INGESTION POLICY
# Specific permissions for S3 data retrieval, Bedrock model invocation, and secret decryption.

resource "aws_iam_policy" "rds_seeder_permissions" {
  name        = "${var.environment}-rds-seeder-permissions"
  description = "Provides the RDS Seeder access to source datasets, embedding models, and credentials."

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogAccess"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Sid      = "S3DatasetRead"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.dataset_storage.arn}/*"
      },
      {
        Sid      = "BedrockEmbeddingInvoke"
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = "arn:aws:bedrock:eu-west-2::foundation-model/amazon.titan-embed-text-v2:0"
      },
      {
        Sid      = "SecretsManagerAccess"
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = data.aws_secretsmanager_secret_version.dept_contacts_db_password.arn
      }
    ]
  })
}

## RDS SEEDER ATTACHMENTS
# Links the ingestion and VPC network policies to the Seeder execution role.

# Attaches the custom data ingestion policy (S3, Bedrock, Secrets).
resource "aws_iam_role_policy_attachment" "rds_seeder_main" {
  role       = aws_iam_role.rds_seeder_role.name
  policy_arn = aws_iam_policy.rds_seeder_permissions.arn
}

# Attaches the managed policy required for private database connectivity.
resource "aws_iam_role_policy_attachment" "rds_seeder_vpc_access" {
  role       = aws_iam_role.rds_seeder_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}
