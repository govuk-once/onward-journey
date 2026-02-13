/**
 * PURPOSE: Secure Credential Management and Environment Secrets.
 * This file manages the metadata and containers for sensitive credentials
 * required by the Onward Journey stack.
 *
 * NOTE: To prevent sensitive values from being stored in the Terraform state
 * file, actual secret values (passwords, API keys, etc.) should be set
 * manually via the AWS Console or CLI after the initial resource creation.
 */

resource "aws_secretsmanager_secret" "dept_contacts_db_password" {
  name        = "${var.environment}-dept-contacts-db-password"
  description = "Password for the ${var.environment} department contacts database"

  lifecycle {
    prevent_destroy = true
  }
}

# The Secret Value (To be populated manually in Console or via CLI for safety)
data "aws_secretsmanager_secret_version" "dept_contacts_db_password" {
  secret_id = aws_secretsmanager_secret.dept_contacts_db_password.id
}

resource "aws_secretsmanager_secret" "genesys_credentials" {
  name        = "${var.environment}-genesys-mcp-credentials"
  description = "OAuth credentials for Genesys Cloud Platform API"
}

# The Secret Value (To be populated manually in Console or via CLI for safety)
resource "aws_secretsmanager_secret_version" "genesys_credentials_val" {
  secret_id = aws_secretsmanager_secret.genesys_credentials.id
  secret_string = jsonencode({
    client_id     = "GENESYS_CLIENT_ID"
    client_secret = "GENESYS_CLIENT_SECRET"
  })
}
