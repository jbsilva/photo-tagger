terraform {
  # OpenTofu, not Terraform: the encryption block below exists in OpenTofu only. Renovate must not
  # bump this: it reads the Terraform releases, whose version numbers OpenTofu does not share.
  required_version = ">= 1.8"

  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 5.28"
    }
  }

  # The state and a saved plan hold the values of the tokens that main.tf creates, so both are
  # encrypted with a key derived from the passphrase. "enforced" refuses to write either one in
  # plain text.
  encryption {
    key_provider "pbkdf2" "passphrase" {
      passphrase = var.state_passphrase
    }
    method "aes_gcm" "passphrase" {
      keys = key_provider.pbkdf2.passphrase
    }
    state {
      method   = method.aes_gcm.passphrase
      enforced = true
    }
    plan {
      method   = method.aes_gcm.passphrase
      enforced = true
    }
  }
}
