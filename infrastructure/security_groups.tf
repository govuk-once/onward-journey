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

  tags = {
    Name = "${var.environment}-vpc-endpoints-sg"
  }
}

# Ingress rule allowing HTTPS traffic from the Orchestrator to VPC Endpoints.
resource "aws_security_group_rule" "allow_orchestrator_to_endpoints" {
  description              = "HTTPS from Orchestrator"
  type                     = "ingress"
  from_port                = 443
  to_port                  = 443
  protocol                 = "tcp"
  security_group_id        = aws_security_group.vpc_endpoints.id
  source_security_group_id = aws_security_group.orchestrator.id
}

# RDS METADATA STORE SECURITY GROUP
# Firewall for the RDS instance hosting department contact metadata.
resource "aws_security_group" "rds_metadata" {
  name        = "${var.environment}-rds-metadata-sg-v2"
  description = "Allows authorized data services to query the Department Contacts database"
  vpc_id      = data.aws_vpc.active.id

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.environment}-rds-metadata-sg-v2"
  }
}

# Allow Tooling & Seeder to reach RDS
resource "aws_security_group_rule" "allow_data_services_to_rds" {
  description              = "PostgreSQL from Seeder and RDS Tool"
  type                     = "ingress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  security_group_id        = aws_security_group.rds_metadata.id
  source_security_group_id = aws_security_group.rds_seeder_sg.id
}

# RDS SEEDER SECURITY GROUP
# Group for the Seeder and RDS Tool Lambda (MCP Server).
resource "aws_security_group" "rds_seeder_sg" {
  name        = "${var.environment}-rds-seeder-sg"
  description = "Allows Data Services to reach RDS and AWS Services"
  vpc_id      = data.aws_vpc.active.id

  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.environment}-rds-seeder-sg"
  }
}

# Ingress rule allowing authorized data services to reach AWS service endpoints.
resource "aws_security_group_rule" "allow_data_services_to_endpoints" {
  description              = "Allow data services to reach Bedrock/SecretsManager endpoints"
  type                     = "ingress"
  from_port                = 443
  to_port                  = 443
  protocol                 = "tcp"
  security_group_id        = aws_security_group.vpc_endpoints.id
  source_security_group_id = aws_security_group.rds_seeder_sg.id
}
