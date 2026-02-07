## MANAGED BEDROCK AGENTCORE CAPABILITIES
# Defines the modular utilities used by the Orchestration Layer. Offloads state management and tool connectivity to AWS-managed services.

## AGENTCORE MEMORY
# Managed storage for chat history and session context.
# LangGraph will call this to persist conversation state.
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


## AGENTCORE GATEWAY TARGET: RDS SEARCH TOOL
# Registers the RDS Tool Lambda as a discoverable tool for the Orchestrator.
resource "aws_bedrockagentcore_gateway_target" "rds_search_tool" {
  name               = "${var.environment}-rds-search-tool"
  gateway_identifier = aws_bedrockagentcore_gateway.tool_interface.gateway_id

  target_configuration {
    mcp {
      lambda {
        lambda_arn = aws_lambda_function.rds_tool.arn
        tool_schema {
          inline_payload {
            name        = "query_department_database"
            description = "Searches the department contacts database using semantic vector search."

            input_schema {
              type        = "object"
              description = "Input for the department contact search tool"

              property {
                name        = "query"
                type        = "string"
                description = "The user query to search for contact info."
                required    = true
              }
            }
          }
        }
      }
    }
  }

  credential_provider_configuration {
    gateway_iam_role {} # Use the Gateway's role to invoke the Lambda
  }
}
