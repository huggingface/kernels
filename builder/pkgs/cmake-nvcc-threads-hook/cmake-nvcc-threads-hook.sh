#!/bin/sh

_setNvccThreadsHook() {
  if [ -z "${nvccThreads}" ] || [ "${nvccThreads}" -ne "${nvccThreads}" ] 2>/dev/null; then
    >&2  echo "Number of nvcc threads is not (correctly) set, setting to 4"
    nvccThreads=4
  fi

  # Ensure that we do not use more threads than build cores.
  nvccThreads=$((NIX_BUILD_CORES < nvccThreads ? NIX_BUILD_CORES : nvccThreads ))

  # Change the number of build cores so that build cores * threads is
  # within bounds.
  export NIX_BUILD_CORES=$(($NIX_BUILD_CORES / nvccThreads))

  appendToVar cmakeFlags -DNVCC_THREADS="${nvccThreads}"
}

preConfigureHooks+=(_setNvccThreadsHook)
