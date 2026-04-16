#!/bin/sh

_checkKernelBuildHook() {
  if [ -z "${doKernelBuildCheck:-}" ]; then
    echo "Skipping kernel build check"
  else
    echo "Checking kernel build"
    kernel-builder check-builds "$out/"
  fi
}

postInstallCheckHooks+=(_checkKernelBuildHook)
