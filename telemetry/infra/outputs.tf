output "d1_database_id" {
  description = "The id for ../wrangler.toml and for CLOUDFLARE_D1_DATABASE_ID. It is not a secret."
  value       = cloudflare_d1_database.telemetry.id
}

output "deploy_token" {
  description = "The value for CI's CLOUDFLARE_API_TOKEN secret."
  value       = cloudflare_account_token.deploy.value
  sensitive   = true
}

output "dashboard_token" {
  description = "The CLOUDFLARE_API_TOKEN for the dashboard and the backfill."
  value       = cloudflare_account_token.dashboard.value
  sensitive   = true
}
