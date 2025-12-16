variable "environment" {
  type        = string
  description = "The name of the environment for this instance of the infrastructure, e.g. 'development', 'staging', or <your initials>"
}

variable "allowed_ip_ranges" {
  type        = list(string)
  description = "A list of IP ranges to allow access to the knowledge base opensearch domain from"
}
