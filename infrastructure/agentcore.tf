/**
 * PURPOSE: Managed Bedrock AgentCore Capabilities.
 * This file defines the modular utilities used by the Orchestration Layer.
 * By using managed Memory and Gateway components, we offload the complexity
 * of state management and tool connectivity to AWS-managed services.
 */

# AgentCore Memory: Managed storage for the agent's chat history and session context.
# This can persist the initial history passed from GOV.UK and will
# maintain the conversation state as the user interacts with Onward Journey.
resource "aws_bedrockagent_memory" "agent_chat_context" {
  memory_name = "${var.environment}-agent-chat-context"

  storage_configuration {
    # SESSION_SUMMARY allows the agent to maintain context over long conversations
    # by distilling turns into concise summaries, essential for Genesys handoffs.
    type = "SESSION_SUMMARY"
  }
}

# AgentCore Gateway: Standardised interface for tool connectivity via MCP.
# This acts as the bridge between the Orchestrator and external data sources.
resource "aws_bedrockagent_gateway" "tool_interface" {
  gateway_name = "${var.environment}-tool-interface"
}
