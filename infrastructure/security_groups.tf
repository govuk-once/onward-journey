/**
 * PURPOSE: Network Security and Traffic Control.
 * This file defines the firewall rules for the Orchestration Layer and its
 * private connectivity to AWS services. All rules should follow the principle of
 * least privilege, restricting traffic to specific security group IDs.
 */

# ORCHESTRATOR SECURITY GROUP
# Controls traffic for the Lambda-based logic layer.
resource "aws_security_group" "orchestrator" {
  name        = "${var.environment}-orchestrator-sg"
  description = "Security group for the Orchestration Layer for environment: ${var.environment}"
  vpc_id      = data.aws_vpc.active.id

  # Allow all outbound traffic for calling Bedrock APIs and querying tool endpoints.
  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.environment}-orchestrator-sg"
  }
}

# VPC ENDPOINTS SECURITY GROUP
# Provides a secure ingress point for the Orchestrator to reach AWS Services.
resource "aws_security_group" "vpc_endpoints" {
  name        = "${var.environment}-vpc-endpoints-sg"
  description = "Private interface for the Orchestrator to reach Bedrock and Secrets Manager"
  vpc_id      = data.aws_vpc.active.id

  # Only traffic originating from the Orchestrator's Security Group is permitted.
  ingress {
    description     = "HTTPS from Orchestrator"
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.orchestrator.id]
  }

  tags = {
    Name = "${var.environment}-vpc-endpoints-sg"
  }
}

# RDS METADATA STORE SECURITY GROUP
resource "aws_security_group" "rds_metadata" {
  name        = "${var.environment}-rds-metadata-sg"
  description = "Allows the Orchestrator to query the Department Contacts database"
  vpc_id      = data.aws_vpc.active.id

  ingress {
    description     = "PostgreSQL from Orchestrator"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.orchestrator.id]
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.environment}-rds-metadata-sg" }
}
