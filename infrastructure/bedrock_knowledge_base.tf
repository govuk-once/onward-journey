resource "aws_bedrockagent_knowledge_base" "main" {
  name     = "knowledge-base"
  role_arn = aws_iam_role.bedrock_kb_role.arn

  knowledge_base_configuration {
    type = "VECTOR"
    vector_knowledge_base_configuration {
      embedding_model_arn = var.embedding_model_arn
    }
  }

  storage_configuration {
    type = "OPENSEARCH_SERVERLESS" # Technically uses the same schema for Provisioned

    # confusingly, for Provisioned domains, you often use the opensearch_serverless_configuration block
    # or the generic configuration depending on provider version.
    # Below is the standard configuration for connecting to a Vector Store.

    opensearch_serverless_configuration {
      collection_arn    = data.aws_opensearch_domain.knowledge_base.arn
      vector_index_name = "bedrock-index"

      field_mapping {
        vector_field   = "bedrock_embedding"
        text_field     = "bedrock_text"
        metadata_field = "bedrock_metadata"
      }
    }
  }
}

resource "aws_iam_role" "bedrock_kb" {
  name = "${var.environment}-bedrock-kb-role"

  assume_role_policy = data.aws_iam_policy_document.bedrock_kb_assume_role_policy.json
}

data "aws_iam_policy_document" "bedrock_kb_assume_role_policy" {
  statement {
    sid     = "AllowBedrockToAssumeRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["bedrock.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "bedrock_kb" {
  role = aws_iam_role.bedrock_kb.name

  policy = data.aws_iam_policy_document.bedrock_kb_role_policy.json
}

data "aws_iam_policy_document" "bedrock_kb_role_policy" {
  statement {
    sid       = "AllowAccessToEmbeddingModel"
    effect    = "Allow"
    resources = [var.embedding_model_arn]
  }

  statement {
    sid    = "AllowDatasetS3Read"
    effect = "Allow"
    actions = [
      "s3:ListBucket",
      "s3:GetObject"
    ]
    resources = [
      aws_s3_bucket.dataset_storage.arn,
      "${aws_s3_bucket.dataset_storage.arn}/*"
    ]
  }

  statement {
    sid    = "AllowOpenSearchWrite"
    effect = "Allow"
    actions = [
      "es:ESHttpGet",
      "es:ESHttpPut",
      "es:ESHttpPost"
    ]
    resources = ["${aws_opensearch_domain.knowledge_base.arn}/*"]
  }
}
