variable "environment" {
  type        = string
  description = "The name of the environment for this instance of the infrastructure, e.g. 'development', 'staging', or <your initials>"
}

variable "allowed_ip_ranges" {
  type        = list(string)
  description = "A list of IP ranges to allow access to the knowledge base opensearch domain from"
}

variable "embedding_model_arn" {
  type        = string
  description = "The ARN of the model to use for embeddings in the knowledge base. Defaults to titan text embeddings v2"
  default     = "arn:aws:bedrock:eu-west-2::foundation-model/amazon.titan-embed-text-v2:0"
}
