terraform {
  required_version = "1.13.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "6.21.0"
    }

    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  # Configured per-environment in environments/<environment name>.config
  backend "s3" {}
}

provider "aws" {
  region = "eu-west-2"

  default_tags {
    tags = {
      Project     = "GOV.UK Agents Onward Journey"
      Environment = var.environment
    }
  }
}
