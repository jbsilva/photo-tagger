variable "account_id" {
  description = "The collector's Cloudflare account. The poe tasks pass CLOUDFLARE_ACCOUNT_ID."
  type        = string
}

variable "zone_name" {
  description = "The zone that serves the collector at telemetry.<zone>."
  type        = string
  default     = "tagger.photo"
}

variable "state_passphrase" {
  description = "The state's passphrase, 16 characters or more. Set TF_VAR_state_passphrase."
  type        = string
  sensitive   = true
}
