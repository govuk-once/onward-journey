output "agent_chat_context_arn" {
  description = "The ARN of the managed chat history store."
  value       = aws_bedrockagentcore_memory.agent_chat_context.arn
}

output "tool_interface_arn" {
  description = "The ARN of the tool gateway for MCP connectivity."
  value       = aws_bedrockagentcore_gateway.tool_interface.gateway_arn
}

output "orchestrator_sg_id" {
  description = "The ID of the security group for the orchestration layer."
  value       = aws_security_group.orchestrator.id
}
