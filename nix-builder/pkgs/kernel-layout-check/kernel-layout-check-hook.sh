#!/bin/sh

echo "Sourcing kernel-layout-check-hook.sh"

kernelLayoutCheckHook() {
  echo "Checking kernel layout"

  if [ -z ${moduleName+x} ]; then
    echo "moduleName must be set in derivation"
    exit 1
  fi

  if [ -z ${framework+x} ]; then
    echo "framework must be set in derivation"
    exit 1
  fi

  if [ "${framework}" = "torch" ]; then
    frameworkSrcDir="torch-ext"
  elif [ "${framework}" = "tvm-ffi" ]; then
    frameworkSrcDir="tvm-ffi-ext"
  else
    echo "Unsupported framework: ${framework}"
    exit 1
  fi

  if [ ! -f source/${frameworkSrcDir}/${moduleName}/__init__.py ]; then
    echo "Python module at source/${frameworkSrcDir}/${moduleName} must contain __init__.py"
    exit 1
  fi

  # TODO: remove once the old location is removed from kernels.
  if [ -e source/${frameworkSrcDir}/${moduleName}/${moduleName} ]; then
    echo "Python module at source/${frameworkSrcDir}/${moduleName} must not have ${moduleName} file or directory."
    exit 1
  fi
}

if [ -z "${dontCheckLayout-}" ]; then
  postUnpackHooks+=(kernelLayoutCheckHook)
fi
