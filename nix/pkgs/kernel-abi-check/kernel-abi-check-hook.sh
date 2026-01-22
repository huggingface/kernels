#!/bin/sh

_checkAbiHook() {
  if [ -z "${doAbiCheck:-}" ]; then
    echo "Skipping ABI check"
  else
    echo "Checking of ABI compatibility"
    find "$out/" -name '*.so' -print0 | \
      xargs -0 -n1 kernel-abi-check
  fi
}

postInstallCheckHooks+=(_checkAbiHook)
