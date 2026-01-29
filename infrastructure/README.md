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

The **Government Department Contacts Store** (RDS PostgreSQL 17.6) is our primary knowledge base. It is enhanced with the `pgvector` extension to support semantic search.

- **The "Empty Secret" Workflow:** To maintain security, we do not store the database password in code or state. Consequently, the initial Terraform deployment for an environment will partially fail as it tries to read a secret version that doesn't exist yet.
- **Accessing the DB:** You must manually set your password in **AWS Secrets Manager** after your first deployment.
- **Secret Name:** Look for `${var.environment}-dept-contacts-db-password` in the AWS Console.
- **Connectivity:** Access is strictly controlled via the `${var.environment}-rds-metadata-sg`. It only accepts inbound traffic from the orchestrator or seeder on port 5432.

### 5. Data Ingestion & Seeder Pipeline

We use a configuration-driven pipeline to populate the RDS database. This is managed via `infrastructure/seed_config.yaml`.

- **CSV to RDS:** Adding a new table is as simple as dropping a CSV into `mock_data/` and adding a table entry into the YAML.
- **Development & Mock Data:** The `interaction_memory.csv` and its YAML entry are provided for testing purposes. You can:
  - Replace the existing file and update its YAML entry.
  - Add entirely new files and YAML entries as required for development.
- **Automatic Vectorization:** If a table in the YAML defines an `embedding` column and `embedding_source_cols`, the Seeder Lambda will automatically call Bedrock Titan v2 to generate vectors for those rows during ingestion.
- **Automatic Triggers:** Terraform monitors the `filemd5` hash of your CSV files and the YAML config. Any change will automatically trigger a targeted Lambda invocation to refresh that specific table.

---

### Troubleshooting & Maintenance

#### When the Seeder Fails

If your data isn't appearing in RDS after an apply, check the following:

1.  **CloudWatch Logs:** Navigate to `/aws/lambda/[initials]-rds-seeder`. Look for "Connection Timeout" or Bedrock "Access Denied" errors.
2.  **VPC Routing:** Ensure the S3 Gateway Endpoint is associated with the private route table. The seeder requires this to fetch the CSV files from S3 without internet access.
3.  **Secrets Format:** Ensure the DB password in Secrets Manager is saved as a **Plaintext** string, not a JSON key-pair.

#### Manual Table Refresh

If you need to force-rebuild a table (e.g., if you manually deleted data or want to re-run the vectorization without changing the CSV), use the following targeted replace command:

```bash
terraform apply -replace='terraform_data.rds_sync_trigger["mock_rag_data.csv"]'
```

---

### Developer Quick Start

1.  **Initialise Terraform:** `terraform init`
2.  **Create/Select your workspace:** `terraform workspace new [initials]` / `terraform workspace select [initials]`
3.  **Validate:** Run `terraform validate` to ensure your specific environment names meet Bedrock's regex requirements.
4.  **First Deployment (Expected Failure):** `terraform apply`
    - **Note:** This will create the Secret container for the database password but will fail at the RDS creation step with a 'couldn't find resource' error for the secret version. This is expected.
5.  **Important:** Navigate to the **AWS Console** -> **Secrets Manager** (Ensure region is `eu-west-2`) and set the value for your `${var.environment}-dept-contacts-db-password` as a **Plaintext** value (no brackets required) so the stack can connect to the database.
6.  **Second Deployment & RDS Data Seeding:** Run `terraform apply` again.
    - Terraform will provision the RDS instance and immediately trigger the Seeder Lambda to populate your tables.
