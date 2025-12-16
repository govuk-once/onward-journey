resource "aws_opensearch_domain" "knowledge_base" {
  domain_name    = "${var.environment}-knowledge-base"
  engine_version = "OpenSearch_3.3"

  cluster_config {
    instance_type  = "t3.medium.search"
    instance_count = 1
  }

  ebs_options {
    ebs_enabled = true
    volume_type = "gp3"
    volume_size = 25 # in GiB
  }

  encrypt_at_rest {
    enabled = true
  }
}

data "aws_iam_policy_document" "knowledge_base_access_policy" {
  statement {
    sid = "AllowAllFromSpecifiedIpRanges"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    effect    = "Allow"
    actions   = ["es:*"]
    resources = ["${aws_opensearch_domain.knowledge_base.arn}/*"]

    condition {
      test     = "IpAddress"
      variable = "aws:SourceIp"
      values   = var.allowed_ip_ranges
    }
  }

  statement {
    sid = "AllowInferenceRoleRead"

    principals {
      type        = "AWS"
      identifiers = [aws_iam_role.inference.arn]
    }

    effect = "Allow"
    actions = [
      "es:Get*",
      "es:List*",
      "es:Describe*"
    ]
    resources = ["${aws_opensearch_domain.knowledge_base.arn}/*"]
  }
}

resource "aws_opensearch_domain_policy" "knowledge_base" {
  domain_name     = aws_opensearch_domain.knowledge_base.domain_name
  access_policies = data.aws_iam_policy_document.knowledge_base_access_policy.json
}
