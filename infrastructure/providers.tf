terraform {
  required_version = "1.13.5"

  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "6.21.0"
    }

    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
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
