terraform {
  required_version = "1.14.1"

  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "6.25.0"
    }
  }

  backend "s3" {
    region = "eu-west-2"

    bucket = var.tfstate_bucket_name
    key = "application.tfstate"
    
    use_lockfile = true
  }
}

provider "aws" {
  region = "eu-west-2"

  default_tags {
    tags = {
      Team = "GOV.UK Agents Onward Journey"
      Module = "application"
      Workspace = terraform.workspace
    }
  }
}
