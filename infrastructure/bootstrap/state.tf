resource "aws_s3_bucket" "remote-state" {
  bucket = "onward-journey-infrastructure-state-${terraform.workspace}"
}

resource "aws_s3_bucket_versioning" "remote-state-versioning" {
  bucket = aws_s3_bucket.remote-state.id
  versioning_configuration {
    status = "Enabled"
  }
}
