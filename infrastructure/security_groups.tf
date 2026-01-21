/**
 * PURPOSE: Traffic Control and Firewall Rules.
 * Defines environment-specific ingress and egress rules for the agent
 * Orchestration Layer.
 */

resource "aws_security_group" "orchestrator" {
  name        = "${var.environment}-orchestrator-sg"
  description = "Security group for the Orchestration Layer for environment: ${var.environment}"
  vpc_id      = data.aws_vpc.active.id

  # Allow all outbound traffic for calling Bedrock APIs and querying tool endpoints.
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.environment}-orchestrator-sg"
  }
}
