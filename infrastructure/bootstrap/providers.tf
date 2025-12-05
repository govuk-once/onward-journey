terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "6.25.0"
    }
  }
}

provider "aws" {
  region = "eu-west-2"
  profile = "made-tech-sandbox"

  default_tags {
    tags = {
      Team = "GOV.UK Agents Onward Journey"
    }
  }
}