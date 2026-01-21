/**
 * PURPOSE: Secure Credential Management for Department Contacts.
 * Manages the database master password. The actual value should be set
 * via the AWS Console or CLI to avoid sensitive data in the state file.
 */

resource "aws_secretsmanager_secret" "dept_contacts_db_password" {
  name        = "${var.environment}-dept-contacts-db-password"
  description = "Master password for the ${var.environment} department contacts database"
}

data "aws_secretsmanager_secret_version" "dept_contacts_db_password" {
  secret_id = aws_secretsmanager_secret.dept_contacts_db_password.id
}
