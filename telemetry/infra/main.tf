# The account-level parts of the telemetry collector: the D1 database that keeps every event, and
# two narrow API tokens. The Worker, its route, and its bindings live in ../wrangler.toml, and the
# table schema in ../migrations. CI deploys both on every push, so this changes rarely.

# The provider reads CLOUDFLARE_API_TOKEN. That token needs D1 Write, Account API Tokens Write, and
# Zone Read on the zone.
provider "cloudflare" {}

data "cloudflare_zone" "collector" {
  filter = { name = var.zone_name }
}

data "cloudflare_account_api_token_permission_groups_list" "all" {
  account_id = var.account_id
}

locals {
  # Permission group ids by name, one map per scope. A few names exist once per scope, so a name
  # alone is ambiguous, and the order the API lists them in can change. one() fails the plan if a
  # name the tokens use is ever ambiguous within its scope too, rather than silently picking a
  # group. A duplicate among the names nothing here uses does not matter.
  groups = data.cloudflare_account_api_token_permission_groups_list.all.result
  account_groups = {
    for group in local.groups : group.name => group.id...
    if contains(group.scopes, "com.cloudflare.api.account")
  }
  zone_groups = {
    for group in local.groups : group.name => group.id...
    if contains(group.scopes, "com.cloudflare.api.account.zone")
  }
  account_permission = {
    for name in ["Workers Scripts Write", "D1 Write", "Account Analytics Read", "D1 Read"] :
    name => one(local.account_groups[name])
  }
  zone_permission = {
    for name in ["Workers Routes Write"] : name => one(local.zone_groups[name])
  }

  account_resources = jsonencode({ "com.cloudflare.api.account.${var.account_id}" = "*" })
  zone_resources = jsonencode({
    "com.cloudflare.api.account.zone.${data.cloudflare_zone.collector.id}" = "*"
  })
}

resource "cloudflare_d1_database" "telemetry" {
  account_id = var.account_id
  name       = "photo-tagger-telemetry"
  # The primary takes every write. Western Europe is where the dashboard runs.
  primary_location_hint = "weur"
  # One copy is enough for a few beacons a day. The API reports this even when it is not set.
  read_replication = { mode = "disabled" }

  lifecycle {
    # Replacing the database deletes every event in it.
    prevent_destroy = true
  }
}

# CI's CLOUDFLARE_API_TOKEN secret: it deploys the Worker, attaches its custom domain, and applies
# the D1 migrations. Nothing else.
resource "cloudflare_account_token" "deploy" {
  account_id = var.account_id
  name       = "photo-tagger-telemetry-deploy"
  policies = [
    {
      effect = "allow"
      permission_groups = [
        for name in ["Workers Scripts Write", "D1 Write"] : { id = local.account_permission[name] }
      ]
      resources = local.account_resources
    },
    {
      effect            = "allow"
      permission_groups = [{ id = local.zone_permission["Workers Routes Write"] }]
      resources         = local.zone_resources
    },
  ]
}

# The dashboard's and the backfill's CLOUDFLARE_API_TOKEN: it reads the events and can write
# nothing.
resource "cloudflare_account_token" "dashboard" {
  account_id = var.account_id
  name       = "photo-tagger-telemetry-dashboard"
  policies = [
    {
      effect = "allow"
      permission_groups = [
        for name in ["Account Analytics Read", "D1 Read"] : { id = local.account_permission[name] }
      ]
      resources = local.account_resources
    },
  ]
}
