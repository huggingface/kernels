#!/bin/sh

echo "Sourcing kernel-layout-check-hook.sh"

kernelLayoutCheckHook() {
  echo "Checking kernel layout"

  if [ -z ${moduleName+x} ]; then
    echo "moduleName must be set in derivation"
    exit 1
  fi

  if [ ! -f source/torch-ext/${moduleName}/__init__.py ]; then
    echo "Python module at source/torch-ext/${moduleName} must contain __init__.py"
    exit 1
  fi

  # TODO: remove once the old location is removed from kernels.
  if [ -e source/torch-ext/${moduleName}/${moduleName} ]; then
    echo "Python module at source/torch-ext/${moduleName} must not have ${moduleName} file or directory."
    exit 1
  fi
}

if [ -z "${dontCheckLayout-}" ]; then
  postUnpackHooks+=(kernelLayoutCheckHook)
fi
