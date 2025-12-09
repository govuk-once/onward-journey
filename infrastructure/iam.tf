resource "aws_iam_role" "onward_journey_inference" {
  name               = "onward-journey-${var.environment}-inference-role"
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

resource "aws_iam_role_policy_attachment" "onward_journey_inference_bedrock_access" {
  role = aws_iam_role.onward_journey_inference.name
  # Use AWS provided "Bedrock Limited Access" policy
  policy_arn = "arn:aws:iam::aws:policy/AmazonBedrockLimitedAccess"
}
