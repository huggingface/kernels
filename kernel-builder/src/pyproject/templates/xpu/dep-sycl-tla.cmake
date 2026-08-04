if(GPU_LANG STREQUAL "SYCL")

find_package(SyclTla)

if(DPCPP_VERSION STREQUAL "2025.3")
  set(SYCL_TLA_REVISION "v0.8" CACHE STRING "sycl-tla revision to use")
elseif(DPCPP_VERSION STREQUAL "2026.0")
  # v0.9.2 == the rev pinned in nix-builder; v0.9.1 has no cri kernel support.
  set(SYCL_TLA_REVISION "v0.9.2" CACHE STRING "sycl-tla revision to use")
else()
  message(FATAL_ERROR "Unknown DPCPP_VERSION: ${DPCPP_VERSION}")
endif()

if (NOT SyclTla_FOUND)
  set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")
  set(CUTLASS_ENABLE_BENCHMARKS OFF CACHE BOOL "Disable sycl-tla Benchmarks")
# Use the specified sycl-tla source directory for compilation if SYCL_TLA_SRC_DIR is provided
  if (DEFINED ENV{SYCL_TLA_SRC_DIR})
    set(SYCL_TLA_SRC_DIR $ENV{SYCL_TLA_SRC_DIR})
  endif()

  if(SYCL_TLA_SRC_DIR)
    if(NOT IS_ABSOLUTE SYCL_TLA_SRC_DIR)
      get_filename_component(SYCL_TLA_SRC_DIR "${SYCL_TLA_SRC_DIR}" ABSOLUTE)
    endif()
    message(STATUS "The SYCL_TLA_SRC_DIR is set, using ${SYCL_TLA_SRC_DIR} for compilation")
    FetchContent_Declare(cutlass SOURCE_DIR ${SYCL_TLA_SRC_DIR})
  else()
    # Speed up sycl-tla download by retrieving only the specified GIT_TAG instead of the history.
    # Important: If GIT_SHALLOW is enabled then GIT_TAG works only with branch names and tags.
    # So if the GIT_TAG above is updated to a commit hash, GIT_SHALLOW must be set to FALSE
    if(SYCL_TLA_REVISION MATCHES "^v")
      set(SYCL_TLA_GIT_SHALLOW TRUE)
    else()
      set(SYCL_TLA_GIT_SHALLOW FALSE)
    endif()
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/intel/sycl-tla.git
        GIT_TAG ${SYCL_TLA_REVISION}
        GIT_PROGRESS TRUE
        GIT_SHALLOW ${SYCL_TLA_GIT_SHALLOW}
    )
  endif()

  # Set Intel backend env
  message(STATUS "Setting Intel GPU optimization env vars for sycl-tla")
  set(CUTLASS_ENABLE_SYCL ON CACHE BOOL "Enable SYCL for sycl-tla")
  add_compile_definitions(CUTLASS_ENABLE_SYCL=1)
  set(DPCPP_SYCL_TARGET "intel_gpu_bmg_g21,intel_gpu_pvc" CACHE STRING "SYCL target for Intel GPU")
  add_compile_definitions(DPCPP_SYCL_TARGET=intel_gpu_bmg_g21,intel_gpu_pvc)
  set(SYCL_INTEL_TARGET ON CACHE BOOL "Enable SYCL for INTEL")
  add_compile_definitions(SYCL_INTEL_TARGET=1)

  set(ENV{SYCL_PROGRAM_COMPILE_OPTIONS} "-ze-opt-large-register-file")
  set(ENV{IGC_VISAOptions} "-perfmodel")
  set(ENV{IGC_VectorAliasBBThreshold} "10000")
  set(ENV{IGC_ExtraOCLOptions} "-cl-intel-256-GRF-per-thread")

  FetchContent_MakeAvailable(cutlass)

  include_directories(${CUTLASS_INCLUDE_DIR})
  include_directories(${CUTLASS_TOOLS_UTIL_INCLUDE_DIR})
else()
  include_directories(${CUTLASS_INCLUDE_DIR})
  include_directories(${CUTLASS_TOOLS_UTIL_INCLUDE_DIR})
endif(NOT SyclTla_FOUND)
if(SYCL_TLA_REVISION MATCHES "^v3\\.9")
  add_compile_definitions(OLD_API=1)
endif()

# --- Fat binary: pvc + bmg + cri AOT images, plus a spir64 JIT fallback ------
# `-device cri` lowers the generic spir64_gen image; pvc/bmg alias images
# self-target. MXFP is cri-only (ocloc aborts it elsewhere), so build.toml
# builds those TUs spir64_gen-only and dispatch gates them on IP version >= 35.
set(SYCL_AOT_DEVICES "cri")
set(SYCL_OFFLOAD_TARGETS "intel_gpu_pvc,intel_gpu_bmg_g21,spir64_gen,spir64")

# sycl-tla needs extra extensions (split-barrier always; block-IO and
# matrix-multiply on newer DPCPP). Since the translator replaces the whole
# -spirv-ext list, we query the driver's defaults and append the extras.
set(_sycl_tla_extra_ext "+SPV_INTEL_split_barrier")
if(DPCPP_VERSION STREQUAL "2025.2" OR DPCPP_VERSION STREQUAL "2025.3" OR DPCPP_VERSION STREQUAL "2026.0" OR SYCL_TLA_REVISION STREQUAL "v0.5")
  string(APPEND _sycl_tla_extra_ext ",+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate")
endif()

set(_sycl_ext_probe "${CMAKE_CURRENT_BINARY_DIR}/_sycl_spirv_ext_probe.cpp")
file(WRITE "${_sycl_ext_probe}" "int main() { return 0; }\n")
# "-###" MUST be quoted: an unquoted # starts a CMake comment and would drop the
# rest of the COMMAND, making icpx read from stdin instead of dumping its flags.
execute_process(
  COMMAND ${ICPX_COMPILER} -fsycl -fsycl-targets=spir64_gen "-###" "${_sycl_ext_probe}"
  OUTPUT_VARIABLE _sycl_ext_probe_out
  ERROR_VARIABLE _sycl_ext_probe_err)
string(REGEX MATCH "-spirv-ext=[^\" ]+" _sycl_default_ext "${_sycl_ext_probe_out}${_sycl_ext_probe_err}")
if(_sycl_default_ext)
  # Strip the "-spirv-ext=-all," prefix, leaving the "+ext,+ext,..." default set.
  string(REGEX REPLACE "^-spirv-ext=(-all,)?" "" _sycl_default_ext "${_sycl_default_ext}")
  set(SYCL_SPIRV_EXT "${_sycl_default_ext},${_sycl_tla_extra_ext}")
else()
  message(WARNING "Could not determine default SPIR-V extensions from ${ICPX_COMPILER}; "
                  "using sycl-tla extras only, which may disable compiler defaults.")
  set(SYCL_SPIRV_EXT "${_sycl_tla_extra_ext}")
endif()

xpu_compose_sycl_flags()

endif(GPU_LANG STREQUAL "SYCL")
