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
      DB_HOST     = aws_db_instance.dept_contacts_metadata.address
      DB_NAME     = aws_db_instance.dept_contacts_metadata.db_name
      DB_USER     = aws_db_instance.dept_contacts_metadata.username
      DB_PASSWORD = data.aws_secretsmanager_secret_version.dept_contacts_db_password.secret_string
      BUCKET_NAME = aws_s3_bucket.dataset_storage.id
    }
  }

  depends_on = [aws_cloudwatch_log_group.rds_seeder]
}
