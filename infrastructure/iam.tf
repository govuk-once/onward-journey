## ORCHESTRATOR INFERENCE ROLE
# Primary execution role for the Onward Journey Orchestration Layer.
resource "aws_iam_role" "inference" {
  name               = "${var.environment}-inference-role"
  assume_role_policy = data.aws_iam_policy_document.allow_all_assume_role.json
}

data "aws_iam_policy_document" "allow_all_assume_role" {
  statement {
    sid = "AllowAllIAMUsersToAssumeRole"

    actions = [
      "sts:AssumeRole"
    ]

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
          # Session management actions
          "bedrock:CreateSession",
          "bedrock:UpdateSession",
          "bedrock:GetSession",
          "bedrock:DeleteSession", # Useful for clean-up logic

          # AgentCore specific actions for Memory/Gateway
          "bedrock:InvokeAgent",
          "bedrock:GetAgentMemory", # Specifically required to pull context back
          "bedrock:UpdateAgentMemory"
        ]
        Resource = [
          aws_bedrockagentcore_memory.agent_chat_context.arn,
          aws_bedrockagentcore_gateway.tool_interface.gateway_arn
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "inference_agentcore" {
  role       = aws_iam_role.inference.name
  policy_arn = aws_iam_policy.agentcore_access.arn
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


# BEDROCK AGENTCORE SERVICE ROLE
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
