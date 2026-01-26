## MANAGED BEDROCK AGENTCORE CAPABILITIES
# Defines the modular utilities used by the Orchestration Layer. Offloads state management and tool connectivity to AWS-managed services.

## AGENTCORE MEMORY
# Managed storage for the agent's chat history and session context. Persists interaction history and maintains conversational state.
resource "aws_bedrockagentcore_memory" "agent_chat_context" {
  name                      = "${var.environment}_agent_chat_context"
  memory_execution_role_arn = aws_iam_role.agentcore_role.arn
  event_expiry_duration     = 30
}

## AGENTCORE GATEWAY
# Standardised interface for tool connectivity via MCP. Acts as the bridge between the Orchestrator and external data sources.
resource "aws_bedrockagentcore_gateway" "tool_interface" {
  name            = "${var.environment}-tool-interface"
  role_arn        = aws_iam_role.agentcore_role.arn
  protocol_type   = "MCP"
  authorizer_type = "AWS_IAM"
}
