/**
 * PURPOSE: Dynamic Network Discovery.
 * This file provides the logic for workspaces to identify and utilise the
 * existing permanent VPC. It checks for provisioned resources by tag;
 * if found, it adopts the existing IDs, ensuring that multiple developers
 * can operate within the same network environment without resource duplication.
 */

# Search for an existing 'main-vpc' created by a previous workspace run.
data "aws_vpcs" "existing" {
  filter {
    name   = "tag:Name"
    values = ["main-vpc"]
  }
}

locals {
  # Logic to determine if foundational networking is already provisioned.
  vpc_exists = length(data.aws_vpcs.existing.ids) > 0
}

# The Active VPC ID used by all workspace-specific resources (e.g., RDS Security Groups).
data "aws_vpc" "active" {
  id = local.vpc_exists ? data.aws_vpcs.existing.ids[0] : aws_vpc.main.id
}

# Discover the 'app-private' tier subnets for placing Lambda and RDS instances.
data "aws_subnets" "private" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.active.id]
  }
  filter {
    name   = "tag:Tier"
    values = ["app-private"]
  }
}

# Find the route tables for the private subnets to attach the S3 Gateway
data "aws_route_tables" "private" {
  vpc_id = data.aws_vpc.active.id

  filter {
    name   = "tag:Name"
    values = ["*-app-pvt-*"]
  }
}
