terraform {
  required_version = "1.14.1"
  
  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "6.25.0"
    }
  }
}

provider "aws" {
  region = "eu-west-2"

  default_tags {
    tags = {
      Team = "GOV.UK Agents Onward Journey"
      Module = "bootstrap"
      Workspace = terraform.workspace
    }
  }
}
