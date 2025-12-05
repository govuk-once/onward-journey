terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "6.25.0"
    }
  }

  backend "s3" {
    region = "eu-west-2"

    bucket = "onward-journey-infrastructure-state"
    key = "application.tfstate"
    
    use_lockfile = true
  }
}

provider "aws" {
  region = "eu-west-2"

  default_tags {
    tags = {
      Team = "GOV.UK Agents Onward Journey"
    }
  }
}
