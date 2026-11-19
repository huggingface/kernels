#!/bin/sh

echo "Sourcing zip-build-hook.sh"

zipBuildHook() {
  echo "Zipping builds with the ZIP build layout"
  @kernel_builder@/bin/kernel-builder zip-build "$NIX_BUILD_TOP/$sourceRoot" $out/
}

if [ -z "${dontZipBuild-}" ]; then
  postFixupHooks+=(zipBuildHook)
fi
