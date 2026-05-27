#!/bin/sh

echo "Sourcing kernel-abi-check-hook.sh"

_checkAbiHook() {
  if [ -z "${doAbiCheck:-}" ]; then
    echo "Skipping ABI check"
  else

    if [ -z "${torchStableAbiVersion:-}" ]; then
      _torchStableAbiFlag=""
    else
      _torchStableAbiFlag="--torch-stable-abi=${torchStableAbiVersion}"
    fi

    echo "Checking of ABI compatibility"
    find "$out/" -name '*.so' -print0 | \
      xargs -0 -n1 kernel-abi-check ${_torchStableAbiFlag}
  fi
}

postInstallCheckHooks+=(_checkAbiHook)
