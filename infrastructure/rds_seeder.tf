## DATABASE SEEDING REGISTRY
# Maps specific source files to their target RDS tables.
# To add a new data source, simply add a key-value pair to this map.

locals {
  data_sources = {
    "mock_rag_data.csv" = "dept_contacts"
    # "future_data.csv" = "future_table"
  }
}

## RE-SEEDING LOGIC
# Monitors file hashes and triggers a targeted Lambda invocation only on change.
resource "terraform_data" "rds_sync_trigger" {
  for_each = local.data_sources

  input = {
    file_name  = each.key
    table_name = each.value
  }

  # Terraform calculates the hash of the local file to detect changes
  triggers_replace = [
    filemd5("${path.module}/../mock_data/${each.key}")
  ]

  # Targeted invocation: passing the specific file and table as a JSON payload
  provisioner "local-exec" {
    command = "aws lambda invoke --function-name ${aws_lambda_function.rds_seeder.function_name} --cli-binary-format raw-in-base64-out --payload '${jsonencode(self.input)}' --region ${var.aws_region} --cli-read-timeout 300 response_${each.value}.json"
  }

  # Ensures S3, RDS and Lambda are fully ready before attempting to seed
  depends_on = [
    aws_db_instance.dept_contacts_metadata,
    aws_s3_object.mock_data_upload,
    aws_lambda_function.rds_seeder
  ]
}
