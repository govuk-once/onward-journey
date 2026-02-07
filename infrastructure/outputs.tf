## AGENTCORE IDENTIFIERS
# These IDs are required for the Orchestrator's environment variables.

output "agent_chat_context_id" {
  description = "The unique identifier for AgentCore Memory."
  value       = aws_bedrockagentcore_memory.agent_chat_context.id
}

output "tool_interface_id" {
  description = "The unique identifier for the AgentCore Gateway."
  value       = aws_bedrockagentcore_gateway.tool_interface.gateway_id
}

## RESOURCE ARNs
# Required for IAM policies and cross-account references.

output "agent_chat_context_arn" {
  description = "The ARN of the managed chat history store."
  value       = aws_bedrockagentcore_memory.agent_chat_context.arn
}

output "tool_interface_arn" {
  description = "The ARN of the tool gateway for MCP connectivity."
  value       = aws_bedrockagentcore_gateway.tool_interface.gateway_arn
}

## NETWORKING
output "orchestrator_sg_id" {
  description = "The ID of the security group for the orchestration layer."
  value       = aws_security_group.orchestrator.id
}
