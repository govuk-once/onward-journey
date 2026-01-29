/**
 * PURPOSE: Private Connectivity (VPC Endpoints).
 * These endpoints allow resources in private subnets to securely
 * communicate with AWS services without traversing the public internet.
 */

# 1. Bedrock Endpoint - Required for LLM inference and embeddings
resource "aws_vpc_endpoint" "bedrock" {
  vpc_id            = data.aws_vpc.active.id
  service_name      = "com.amazonaws.${var.aws_region}.bedrock-runtime"
  vpc_endpoint_type = "Interface"

  # Deploying into the private application subnets
  subnet_ids          = data.aws_subnets.private.ids
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true

  tags = { Name = "${var.environment}-bedrock-endpoint" }
}

# 2. Secrets Manager Endpoint - Required to retrieve the DB password
resource "aws_vpc_endpoint" "secrets" {
  vpc_id            = data.aws_vpc.active.id
  service_name      = "com.amazonaws.${var.aws_region}.secretsmanager"
  vpc_endpoint_type = "Interface"

  subnet_ids          = data.aws_subnets.private.ids
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true

  tags = { Name = "${var.environment}-secrets-endpoint" }
}

# 3. S3 Endpoint (Gateway) - Required to ingest the contacts CSV
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = data.aws_vpc.active.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"

  # Gateway endpoints are "routed" via the route table
  route_table_ids = local.private_route_table_ids

  tags = { Name = "${var.environment}-s3-endpoint" }
}
