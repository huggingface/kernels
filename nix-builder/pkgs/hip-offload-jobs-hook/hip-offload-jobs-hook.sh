#!/bin/sh

_setHipOffloadJobsHook() {
  if [ -z "${hipOffloadJobs}" ] || [ "${hipOffloadJobs}" -ne "${hipOffloadJobs}" ] 2>/dev/null; then
    >&2  echo "Number of HIP offload jobs is not (correctly) set, setting to 4"
    hipOffloadJobs=4
  fi

  # Ensure that we do not use more jobs than build cores.
  hipOffloadJobs=$((NIX_BUILD_CORES < hipOffloadJobs ? NIX_BUILD_CORES : hipOffloadJobs ))

  # Change the number of build cores so that build cores * jobs is
  # within bounds.
  export NIX_BUILD_CORES=$(($NIX_BUILD_CORES / hipOffloadJobs))

  # Picked up by the ROCm clang wrapper to run the per-architecture device
  # compilation jobs in parallel.
  export HIP_OFFLOAD_JOBS="${hipOffloadJobs}"
}

preConfigureHooks+=(_setHipOffloadJobsHook)
