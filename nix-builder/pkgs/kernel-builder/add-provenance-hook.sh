#!/bin/sh

echo "Sourcing add-provenance-hook.sh"

# Add provenance to the metadata of the built variants. Enabled by setting
# `provenance` to a JSON file with the git state of the kernel (or SDK) source
# and of the `kernel-builder` source, with `null` for an unknown state:
#
#   {"kernel": {"commit": "<SHA>", "dirty": false}, "builder": null}
addProvenanceHook() {
  echo "Writing provenance to metadata"

  kernel-builder add-provenance "${provenance}" "$out/"
}

if [ -n "${provenance:-}" ]; then
  appendToVar preDistPhases addProvenanceHook
fi
