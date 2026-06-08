#!/bin/sh

echo "Sourcing hash-kernel-hook.sh"

hashKernelHook() {
  echo "Writing kernel hashes to metadata"
  @kernel_builder@/bin/kernel-builder hash $out/
}

if [ -z "${dontHashKernel-}" ]; then
  appendToVar preDistPhases hashKernelHook
fi
