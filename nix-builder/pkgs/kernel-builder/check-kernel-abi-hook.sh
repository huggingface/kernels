#!/bin/sh

echo "Sourcing check-kernel-abi-hook.sh"

_checkAbiHook() {
  if [ -z "${doAbiCheck:-}" ]; then
    echo "Skipping ABI check"
  else

    echo "Checking of ABI compatibility"

    # `pythonAbiVersion` is the minimum Python version for the stable ABI
    # (abi3), `torchStableAbiVersion` is the Torch stable ABI version. Both
    # are optional.
    local args=()
    if [ -n "${pythonAbiVersion:-}" ]; then
      args+=("--python-abi=${pythonAbiVersion}")
    fi
    if [ -n "${torchStableAbiVersion:-}" ]; then
      args+=("--torch-stable-abi=${torchStableAbiVersion}")
    fi

    kernel-builder check-abi ${args[@]+"${args[@]}"} "$out/"
  fi
}

postInstallCheckHooks+=(_checkAbiHook)
