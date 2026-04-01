#!/bin/bash
set -euo pipefail

# kernel-builder installer
# Usage: curl -fsSL https://raw.githubusercontent.com/huggingface/kernels/main/install.sh | bash

FLAKE_REF="github:huggingface/kernels"
NIX_PROFILE_SCRIPT="/nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh"

# --- Colors (respect NO_COLOR) ---

if [ -z "${NO_COLOR:-}" ] && [ -t 1 ]; then
  BOLD="\033[1m"
  GREEN="\033[0;32m"
  YELLOW="\033[0;33m"
  RED="\033[0;31m"
  RESET="\033[0m"
else
  BOLD=""
  GREEN=""
  YELLOW=""
  RED=""
  RESET=""
fi

info()  { echo -e "${BOLD}${GREEN}==>${RESET} ${BOLD}$1${RESET}"; }
warn()  { echo -e "${BOLD}${YELLOW}warning:${RESET} $1"; }
error() { echo -e "${BOLD}${RED}error:${RESET} $1" >&2; }

# --- macOS: Xcode check ---

check_xcode() {
  if [ "$(uname -s)" = "Darwin" ]; then
    if ! xcode-select -p &>/dev/null; then
      warn "Xcode is not installed. It is required for building Metal kernels."
      echo "  Install it with: xcode-select --install"
    fi
  fi
}

# --- Nix ---

find_nix() {
  if command -v nix &>/dev/null; then
    return 0
  elif [ -x "/nix/var/nix/profiles/default/bin/nix" ]; then
    export PATH="/nix/var/nix/profiles/default/bin:$PATH"
    return 0
  fi
  return 1
}

install_nix() {
  if find_nix; then
    info "Nix is already installed: $(nix --version)"
    return 0
  fi

  info "Installing Determinate Nix..."
  curl -fsSL https://install.determinate.systems/nix | sh -s -- install --no-confirm

  # Source the Nix profile so nix is available in this shell.
  if [ -f "$NIX_PROFILE_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$NIX_PROFILE_SCRIPT"
  fi

  if ! find_nix; then
    error "Nix installation completed but 'nix' was not found in PATH."
    echo "  Try restarting your shell or running:"
    echo "    . $NIX_PROFILE_SCRIPT"
    exit 1
  fi

  info "Nix installed: $(nix --version)"
}

# --- Binary cache ---

HF_SUBSTITUTER="https://huggingface.cachix.org"
HF_PUBLIC_KEY="huggingface.cachix.org-1:ynTPbLS0W8ofXd9fDjk1KvoFky9K2jhxe6r4nXAkc/o="

configure_cache() {
  if nix show-config 2>/dev/null | grep -q "huggingface.cachix.org"; then
    info "Hugging Face binary cache is already configured"
    return 0
  fi

  info "Configuring Hugging Face binary cache..."
  local user
  user="$(whoami)"

  # 'extra-trusted-users' appends to the existing trusted-users list.
  # Using 'trusted-users' would override the default (root), which could
  # break the Nix installation.
  sudo tee -a /etc/nix/nix.conf >/dev/null <<EOF
extra-trusted-users = $user
extra-substituters = $HF_SUBSTITUTER
extra-trusted-public-keys = $HF_PUBLIC_KEY
EOF
  sudo systemctl restart nix-daemon 2>/dev/null || sudo pkill -HUP nix-daemon || true
  sleep 3
  info "Binary cache configured"
}

# --- Install kernel-builder ---

install_kernel_builder() {
  info "Installing kernel-builder..."

  local nix_args=(--accept-flake-config)

  # macOS requires relaxed sandboxing to access the Metal compiler.
  if [ "$(uname -s)" = "Darwin" ]; then
    nix_args+=(--extra-conf "sandbox = relaxed")
  fi

  nix profile add "${nix_args[@]}" "${FLAKE_REF}#kernel-builder"

  info "kernel-builder installed: $(kernel-builder --version)"
}

# --- Main ---

main() {
  echo ""
  echo -e "${BOLD}kernel-builder installer${RESET}"
  echo ""

  check_xcode
  install_nix
  configure_cache
  install_kernel_builder

  echo ""
  echo -e "${BOLD}${GREEN}kernel-builder installed successfully!${RESET}"
  echo ""
  echo "  Next steps:"
  echo "    1. Create a new kernel:     kernel-builder init my-kernel"
  echo "    2. Build your kernel:       cd my-kernel && nix run .#build-and-copy -L"
  echo "    3. Read the docs:           https://huggingface.co/docs/kernels/"
  echo ""
  echo "  To update kernel-builder later:"
  echo "    nix profile upgrade --all"
  echo ""
  echo "  Note: you may need to restart your shell or run:"
  echo "    . $NIX_PROFILE_SCRIPT"
  echo ""
}

main
