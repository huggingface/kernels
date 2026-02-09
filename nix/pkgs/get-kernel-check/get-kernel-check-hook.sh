#!/bin/sh

echo "Sourcing get-kernel-check-hook.sh"

_getKernelCheckHook() {
  echo "Checking loading kernel with get_kernel"

  if [ -z ${moduleName+x} ]; then
    echo "moduleName must be set in derivation"
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

  # Some kernels want to write stuff (especially when they use Triton).
  HOME=$(mktemp -d -t test.XXXXXX) || exit 1
  trap "rm -rf '$HOME'" EXIT

  PYTHONPATH="@kernels@" \
    @python3@ -c "from pathlib import Path; import kernels; kernels.get_local_kernel(Path('${out}'), '${moduleName}')"
}

postInstallCheckHooks+=(_getKernelCheckHook)
