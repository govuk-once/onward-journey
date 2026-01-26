/**
 * PURPOSE: The Government Department Contacts Store (RDS PostgreSQL).
 * This database acts as the central repository for department and division
 * contact metadata and routing logic. It is deployed into the private
 * application tier to ensure data security and isolation.
 */

resource "aws_db_instance" "dept_contacts_metadata" {
  # Identifier for the AWS Console
  identifier        = "${var.environment}-dept-contacts-metadata"
  allocated_storage = 20
  engine            = "postgres"
  engine_version    = "17.6"
  instance_class    = "db.t4g.micro"
  db_name           = "gov_dept_contacts"
  username          = "onward_admin"

  # Reference the specific password from Secrets Manager
  # TODO: Add password to Secrets Manager
  password = data.aws_secretsmanager_secret_version.dept_contacts_db_password.secret_string

  # Network and Security placement
  db_subnet_group_name   = aws_db_subnet_group.dept_contacts_subnets.name
  vpc_security_group_ids = [aws_security_group.rds_metadata.id]

  # Lifecycle management: 'true' allows for faster teardown during PoC development
  # by skipping the final backup snapshot usually required by AWS.
  skip_final_snapshot = true
  publicly_accessible = false
  storage_encrypted   = true

  tags = {
    Name = "${var.environment}-dept-contacts-metadata"
  }
}

resource "aws_db_subnet_group" "dept_contacts_subnets" {
  name       = "${var.environment}-dept-contacts-subnets"
  subnet_ids = data.aws_subnets.private.ids

  tags = {
    Name = "${var.environment}-dept-contacts-subnets"
  }
}
