#!/bin/bash

# Script name: dev-setup.sh
# Description: set up a photo-tagger development environment.
#
# Idempotent: safe to re-run. It installs uv and ExifTool if missing, creates the virtual
# environment, syncs the dev + test dependency groups from uv.lock, and installs the git hooks.
# Used by the dev container's postCreateCommand, and works as a one-shot bootstrap on a host too.
#
# Override behavior with environment variables, for example:
#   SKIP_STEPS="exiftool" bash scripts/dev-setup.sh   # skip the ExifTool install step
#   INSTALL_HOOKS=false bash scripts/dev-setup.sh      # do not install git hooks

# Default configuration (can be overridden by environment variables)
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_HOOKS="${INSTALL_HOOKS:-true}"
SKIP_STEPS="${SKIP_STEPS:-}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Restrict curl to https and TLS 1.2+ for every download in this script.
readonly CURL_PROTO="=https"

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
    return 0
}
warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    return 0
}
error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

# Run a command with sudo only when not already root and sudo exists. Lets the script install
# system packages both inside the dev container (non-root "vscode" user with sudo) and on a host.
maybe_sudo() {
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
    elif command -v sudo &>/dev/null; then
        sudo "$@"
    else
        error "Need root to run: $*. Re-run as root or install sudo."
    fi
}

# Install the uv package manager if it is not already on PATH.
setup_uv() {
    if [[ "$SKIP_STEPS" == *"uv"* ]]; then
        log "Skipping uv setup"
        return
    fi

    if command -v uv &>/dev/null; then
        log "uv is already installed ($(uv --version))."
        return
    fi

    log "uv is not installed. Installing now..."
    if command -v curl &>/dev/null; then
        curl --proto "$CURL_PROTO" --tlsv1.2 -sSf https://astral.sh/uv/install.sh | sh
    elif command -v wget &>/dev/null; then
        wget --max-redirect=0 -qO- https://astral.sh/uv/install.sh | sh
    else
        error "Please install curl or wget so uv can be downloaded."
    fi

    # The installer drops uv in ~/.local/bin; make it visible for the rest of this script.
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv &>/dev/null || error "uv installation failed."
    log "uv has been installed ($(uv --version))."
}

# Install ExifTool, the binary that pyexiftool drives to read and write photo metadata.
setup_exiftool() {
    if [[ "$SKIP_STEPS" == *"exiftool"* ]]; then
        log "Skipping ExifTool setup"
        return
    fi

    if command -v exiftool &>/dev/null; then
        log "ExifTool is already installed ($(exiftool -ver))."
        return
    fi

    log "ExifTool is not installed. Installing now..."
    if command -v apt-get &>/dev/null; then
        maybe_sudo apt-get update
        maybe_sudo apt-get install -y --no-install-recommends libimage-exiftool-perl
    elif command -v brew &>/dev/null; then
        brew install exiftool
    elif command -v dnf &>/dev/null; then
        maybe_sudo dnf install -y perl-Image-ExifTool
    else
        warn "Could not auto-install ExifTool. Install it manually: https://exiftool.org/"
        return
    fi
    command -v exiftool &>/dev/null && log "ExifTool installed ($(exiftool -ver))."
}

# Create the virtual environment and sync the dev + test dependency groups from uv.lock.
setup_python_env() {
    if [[ "$SKIP_STEPS" == *"deps"* ]]; then
        log "Skipping Python dependency setup"
        return
    fi

    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating virtual environment in $VENV_DIR..."
        uv venv "$VENV_DIR"
    fi

    log "Syncing dependencies (dev + test groups) from uv.lock..."
    uv sync --group dev --group test
}

# Install the git hooks defined in .pre-commit-config.yaml. Run through "uv run" so the local
# hooks (zuban, pycroscope) find their tools inside the synced virtual environment.
setup_hooks() {
    if [[ "$INSTALL_HOOKS" != "true" || "$SKIP_STEPS" == *"hooks"* ]]; then
        log "Skipping git hook installation"
        return
    fi

    log "Installing git hooks with prek..."
    uv run --with prek prek install || warn "Hook installation failed."
}

main() {
    log "Setting up the photo-tagger development environment..."
    setup_uv
    setup_exiftool
    setup_python_env
    setup_hooks
    log "Setup complete. Try: uv run photo-tagger --help"
    return 0
}

main
