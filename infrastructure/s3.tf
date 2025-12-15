resource "aws_s3_bucket" "dataset_storage" {
  bucket = "onward-journey-${var.environment}-datasets"

  # Allow terraform to delete files when destroying for easy environment teardown
  # Dataset files get uploaded when creating a new environment
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "dataset_storage" {
  bucket = aws_s3_bucket.dataset_storage.id

  versioning_configuration {
    status = "Enabled"
  }
}
