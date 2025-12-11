variable "environment" {
  type        = string
  description = "The name of the environment for this instance of the infrastructure, e.g. 'development', 'staging', or <your initials>"
}

variable "aws_account_id" {
  type        = string
  description = "The AWS account ID where this is being deployed"

  validation {
    condition     = can(regex("^\\d{12}$", var.aws_account_id))
    error_message = "The AWS account ID should be a 12-digit number"
  }
}
