#!/bin/sh

echo "Sourcing check-kernel-abi-hook.sh"

_checkAbiHook() {
  if [ -z "${doAbiCheck:-}" ]; then
    echo "Skipping ABI check"
  else

    echo "Checking of ABI compatibility"
    if [ -z "${torchStableAbiVersion:-}" ]; then
      kernel-builder check-abi "$out/"
    else
      kernel-builder check-abi --torch-stable-abi="${torchStableAbiVersion}" "$out/"
    fi
  fi
}

postInstallCheckHooks+=(_checkAbiHook)
