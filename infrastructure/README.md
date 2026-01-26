# Onward Journey: Infrastructure Overview

This directory contains the terraform code to deploy an instance of the onwards journey application and supporting services.

### The "Shared VPC, Isolated Stack" Model

We utilise a model where foundational networking is **shared** across the account, while application resources are **isolated** per developer environment.

- **How:** A single, persistent **Shared VPC** is provisioned once. Individual developer resources (Orchestrators, Databases, etc.) are then deployed into this VPC using Terraform workspaces.
- **Why:** This approach avoids hitting AWS VPC limits and reduces deployment times by reusing the heavy networking infrastructure. It ensures a consistent network configuration across the team while maintaining total security and data isolation for each developer.

---

### 1. The Shared VPC & Subnets

To simplify connectivity and manage quotas, the networking layer is **shared and persistent**.

- **Shared Resources:** `main-vpc`, `app-pvt-2a/b` (Private), and `dmz-pub-2a` (Public).
- **VPC Endpoints:** We provision interface and gateway endpoints for **Bedrock**, **Secrets Manager**, and **S3**. This allows the isolated stack to communicate with AWS services securely without leaving the private network.
- **Protection:** These resources use the `prevent_destroy = true` lifecycle guardrail to prevent accidental network disruption.
- **⚠️ Tear Down Warning:** If you must destroy the VPC, you must first manually comment out the `prevent_destroy` lines in `infrastructure/vpc.tf`. **Do not do this unless you are certain no other developer is using the network.**

### 2. Environment Isolation

We use the `var.environment` variable (set in your `local.auto.tfvars`) to prefix resources. This ensures that your work does not conflict with others.

- **Unique Resources:** IAM Roles, Security Groups, RDS Instances, and Bedrock AgentCore modules.
- **Naming Convention:** `[initials]-[resource-name]` (e.g., `ab-orchestrator-sg`).

### 3. Managed Agent Capabilities (AgentCore)

We leverage Amazon Bedrock’s managed "AgentCore" to handle the complexities of the agentic loop:

- **`agent_chat_context`**: This is the managed memory store. It can persist the history handed over from GOV.UK chat and will maintain the ongoing conversation state during the Onward Journey.
- **`tool_interface`**: A standardised gateway using the Model Context Protocol (MCP) that allows the Orchestrator to securely query tools (like the Department Contacts database). **Note:** Inbound traffic is secured via AWS_IAM (SigV4) authentication.

**⚠️ Naming Note:** Due to inconsistent AWS validation rules, Memory resources use underscores (e.g., `agent_chat_context`) while Gateway resources use hyphens (e.g., `tool-interface`). Terraform handles this via the agentcore.tf file, but be mindful if renaming.

### 4. Database & Secrets Management

The **Government Department Contacts Store** (RDS PostgreSQL) is where we store department and division metadata.

- **Accessing the DB:** You must manually set your password in **AWS Secrets Manager** after your first deployment.
- **Secret Name:** Look for `${var.environment}-dept-contacts-db-password` in the AWS Console.
- **Connectivity:** Access is strictly controlled via the `${var.environment}-rds-metadata-sg`. It only accepts inbound traffic from the orchestrator on port 5432.

---

### Developer Quick Start

1.  Initialise Terraform: `terraform init`
2.  Create/Select your workspace: `Terraform workspace new [initials]` / `terraform workspace select [initials]`
3.  Validate: Run `terraform validate` to ensure your specific environment names meet Bedrock's regex requirements.
4.  Deploy: `terraform apply`
5.  **Important:** Navigate to the **AWS Console** -> **Secrets Manager** and set the value for your `dept-contacts-db-password` so the stack can connect to the database.
