#!/bin/sh

echo "Sourcing torch-ops-check-hook.sh"


pythonOpsCheckHook() {
  echo "Checking kernel layout"

  @python3@/bin/python @hook@ source
}

if [ -z "${dontCheckPythonOps-}" ]; then
  postUnpackHooks+=(pythonOpsCheckHook)
fi
