#!/bin/sh

_setXpuParallelHook() {
  if [ -z "${xpuParallelJobs}" ] || [ "${xpuParallelJobs}" -ne "${xpuParallelJobs}" ] 2>/dev/null; then
    >&2  echo "Number of XPU parallel jobs is not (correctly) set, setting to 12"
    xpuParallelJobs=12
  fi

  # Cap parallel jobs to available cores.
  xpuParallelJobs=$((NIX_BUILD_CORES < xpuParallelJobs ? NIX_BUILD_CORES : xpuParallelJobs))

  # Reduce NIX_BUILD_CORES to limit ninja parallelism for XPU builds,
  export NIX_BUILD_CORES="${xpuParallelJobs}"

  >&2 echo "XPU parallel hook: limiting ninja to -j${NIX_BUILD_CORES}"
}

preConfigureHooks+=(_setXpuParallelHook)
