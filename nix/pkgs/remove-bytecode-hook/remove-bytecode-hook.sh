#!/bin/sh

echo "Sourcing remove-bytecode-hook.sh"

removeBytecodeHook() {
  echo "Removing Python bytecode"
  find $out -type d -name '__pycache__' -exec rm -rf {} +
}

if [ -z "${dontRemoveBytecode-}" ]; then
  appendToVar preDistPhases removeBytecodeHook
fi
