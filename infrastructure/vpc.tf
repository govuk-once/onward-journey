/**
 * PURPOSE: Foundations for the Onward Journey Shared Network.
 * This file defines the core VPC and Subnets that act as the permanent shell
 * for all developer environments. By using 'prevent_destroy', we ensure
 * foundational networking remains stable even when local workspaces are torn down.
 */

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  lifecycle {
    # Guardrail: Foundations are permanent. To delete, comment this line out first.
    prevent_destroy = true
  }

  tags = {
    Name = "main-vpc"
  }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = "eu-west-2${count.index == 0 ? "a" : "b"}"

  lifecycle {
    # Guardrail: Prevents accidental loss of RDS or compute subnets.
    prevent_destroy = true
  }

  tags = {
    Name = "app-pvt-2${count.index == 0 ? "a" : "b"}"
    Tier = "app-private" # Used by vpc_lookup.tf to identify targets for Lambda/RDS.
  }
}

resource "aws_subnet" "public" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.100.0/24"
  availability_zone = "eu-west-2a"

  lifecycle {
    prevent_destroy = true
  }

  tags = {
    Name = "dmz-pub-2a"
    Tier = "dmz-public" # Reserved for NAT Gateway and public-facing ingress components.
  }
}
