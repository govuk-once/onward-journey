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

resource "aws_iam_role_policy_attachment" "inference_bedrock_access" {
  role = aws_iam_role.inference.name
  # Use AWS provided "Bedrock Limited Access" policy
  policy_arn = "arn:aws:iam::aws:policy/AmazonBedrockLimitedAccess"
}

resource "aws_iam_policy" "dataset_read" {
  name        = "${var.environment}-dataset-read"
  description = "Allow read access to the dataset s3 bucket"
  policy      = data.aws_iam_policy_document.dataset_read.json
}

data "aws_iam_policy_document" "dataset_read" {
  statement {
    actions = ["s3:ListBucket"]

    resources = [aws_s3_bucket.dataset_storage.arn]
  }

  statement {
    actions = ["s3:GetObject"]

    resources = ["${aws_s3_bucket.dataset_storage.arn}/*"]
  }
}

resource "aws_iam_role_policy_attachment" "inference_allow_dataset_read" {
  role       = aws_iam_role.inference.name
  policy_arn = aws_iam_policy.dataset_read.arn
}
