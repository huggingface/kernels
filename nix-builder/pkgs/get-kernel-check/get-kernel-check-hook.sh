#!/bin/sh

echo "Sourcing get-kernel-check-hook.sh"

_getKernelCheckHook() {
  echo "Checking loading kernel with get_kernel"

  if [ -z ${moduleName+x} ]; then
    echo "moduleName must be set in derivation"
    exit 1
  fi

  if [ -z ${kernelDeps+x} ]; then
    echo "kernelDeps must be set in derivation"
    exit 1
  fi

  if [ -z ${variant+x} ]; then
    echo "variant must be set in derivation"
    exit 1
  fi

  echo "Check whether the kernel can be loaded with get-kernel: ${moduleName}"

  # We strip the full library paths from the extension. Unfortunately,
  # in a Nix environment, the library dependencies cannot be found
  # anymore. So we have to add the Torch library directory to the
  # dynamic linker path to get it to pick it up.
  if [ $(uname -s) == "Darwin" ]; then
    TORCH_DIR=$(python -c "from pathlib import Path; import torch; print(Path(torch.__file__).parent)")
    export DYLD_LIBRARY_PATH="${TORCH_DIR}/lib:${DYLD_LIBRARY_PATH}"
  fi

  HOME=$(mktemp -d -t test.XXXXXX) || exit 1
  trap "rm -rf '$HOME'" EXIT

  # Prepare fake /sys for tcmalloc. Without this path, tcmalloc will crash:
  #
  # https://github.com/google/tcmalloc/issues/245
  #
  # tcmalloc is used by the TPU libraries.
  local prootCmd=""
  if [[ -n "@useFakeSys@" ]]; then
      echo "Faking /sys for tcmalloc"
      local fakeSys
      fakeSys="$(mktemp -d)"
      trap 'rm -rf -- "${fakeSys}"' EXIT
      mkdir -p "${fakeSys}/devices/system/cpu"
      echo "0-1" > "${fakeSys}/devices/system/cpu/possible"
      prootCmd="@proot@ -b ${fakeSys}:/sys"
  fi

  PYTHONPATH="@kernels@" \
    ${prootCmd} \
      @python3@ @pyhook@
}

postInstallCheckHooks+=(_getKernelCheckHook)
