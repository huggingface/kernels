#!/bin/sh

_setHipccThreadsHook() {
  if [ -z "${hipccThreads}" ] || [ "${hipccThreads}" -ne "${hipccThreads}" ] 2>/dev/null; then
    >&2  echo "Number of HIP compiler threads is not (correctly) set, setting to 4"
    hipccThreads=4
  fi

  # Ensure that we do not use more jobs than build cores.
  hipccThreads=$((NIX_BUILD_CORES < hipccThreads ? NIX_BUILD_CORES : hipccThreads ))

  # Change the number of build cores so that build cores * jobs is
  # within bounds.
  export NIX_BUILD_CORES=$(($NIX_BUILD_CORES / hipccThreads))

  # Picked up by the ROCm clang wrapper to run the per-architecture device
  # compilation jobs in parallel.
  export HIPCC_THREADS="${hipccThreads}"
}

preConfigureHooks+=(_setHipccThreadsHook)
