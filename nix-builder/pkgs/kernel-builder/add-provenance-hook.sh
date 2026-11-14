#!/bin/sh

echo "Sourcing add-provenance-hook.sh"

# Add provenance to the metadata of the built variants. Enabled by
# `doAddProvenance`. `kernelSha` is the full commit of the kernel (or SDK)
# source and `kernelDirty` is set when it has uncommitted changes. Both are
# optional, the kernel provenance is omitted without `kernelSha`.
addProvenanceHook() {
  echo "Writing provenance to metadata"

  local args=()
  if [ -n "${kernelSha:-}" ]; then
    args+=("--kernel-sha=${kernelSha}")
    if [ -n "${kernelDirty:-}" ]; then
      args+=("--kernel-dirty")
    fi
  fi

  kernel-builder add-provenance ${args[@]+"${args[@]}"} "$out/"
}

if [ -n "${doAddProvenance:-}" ]; then
  appendToVar preDistPhases addProvenanceHook
fi
