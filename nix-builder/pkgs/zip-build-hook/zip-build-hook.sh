#!/bin/sh

echo "Sourcing zip-build-hook.sh"

zipBuildHook() {
  echo "Zipping builds with the ZIP build layout"
  @kernel_builder@/bin/kernel-builder zip-build $out/
}

if [ -z "${dontZipBuild-}" ]; then
  postFixupHooks+=(zipBuildHook)
fi
