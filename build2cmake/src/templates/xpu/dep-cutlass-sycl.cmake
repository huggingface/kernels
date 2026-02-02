if(GPU_LANG STREQUAL "SYCL")

find_package(CutlassSycl)

if(DPCPP_VERSION STREQUAL "2025.3")
  set(CUTLASS_SYCL_REVISION "14055e78510b8776ba739755eb57e592fdceefdb" CACHE STRING "CUTLASS revision to use")
elseif(DPCPP_VERSION STREQUAL "2025.2")
  set(CUTLASS_SYCL_REVISION "14055e78510b8776ba739755eb57e592fdceefdb" CACHE STRING "CUTLASS revision to use")
elseif(DPCPP_VERSION STREQUAL "2025.1")
  set(CUTLASS_SYCL_REVISION "v3.9-0.3" CACHE STRING "CUTLASS revision to use")
elseif(DPCPP_VERSION STREQUAL "2025.0")
  set(CUTLASS_SYCL_REVISION "v3.9-0.2" CACHE STRING "CUTLASS revision to use")
else()
  message(FATAL_ERROR "Unknown DPCPP_VERSION: ${DPCPP_VERSION}")
endif()

if (NOT CutlassSycl_FOUND)
  set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")
  set(CUTLASS_ENABLE_BENCHMARKS OFF CACHE BOOL "Disable CUTLASS Benchmarks")
# Use the specified CUTLASS source directory for compilation if CUTLASS_SYCL_SRC_DIR is provided
  if (DEFINED ENV{CUTLASS_SYCL_SRC_DIR})
    set(CUTLASS_SYCL_SRC_DIR $ENV{CUTLASS_SYCL_SRC_DIR})
  endif()

  if(CUTLASS_SYCL_SRC_DIR)
    if(NOT IS_ABSOLUTE CUTLASS_SYCL_SRC_DIR)
      get_filename_component(CUTLASS_SYCL_SRC_DIR "${CUTLASS_SYCL_SRC_DIR}" ABSOLUTE)
    endif()
    message(STATUS "The CUTLASS_SYCL_SRC_DIR is set, using ${CUTLASS_SYCL_SRC_DIR} for compilation")
    FetchContent_Declare(cutlass SOURCE_DIR ${CUTLASS_SYCL_SRC_DIR})
  else()
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/intel/sycl-tla.git
        GIT_TAG ${CUTLASS_SYCL_REVISION}
        GIT_PROGRESS TRUE

        # Speed up CUTLASS download by retrieving only the specified GIT_TAG instead of the history.
        # Important: If GIT_SHALLOW is enabled then GIT_TAG works only with branch names and tags.
        # So if the GIT_TAG above is updated to a commit hash, GIT_SHALLOW must be set to FALSE
        GIT_SHALLOW $<IF:$<MATCHES:${CUTLASS_SYCL_REVISION},^v>,TRUE,FALSE>
    )
  endif()

  # Set Intel backend env
  message(STATUS "Setting Intel GPU optimization env vars for Cutlass-SYCL")
  set(CUTLASS_ENABLE_SYCL ON CACHE BOOL "Enable SYCL for CUTLASS")
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
endif(NOT CutlassSycl_FOUND)
if(CUTLASS_SYCL_REVISION MATCHES "^v3\\.9")
  add_compile_definitions(OLD_API=1)
endif()

string(REPLACE "-fsycl-targets=spir64_gen,spir64" "-fsycl-targets=spir64" sycl_link_flags "${sycl_link_flags}")
string(REPLACE "-device pvc,xe-lpg,ats-m150" "-device bmg_g21,pvc" sycl_link_flags "${sycl_link_flags}")
string(APPEND sycl_link_flags "-Xspirv-translator;-spirv-ext=+SPV_INTEL_split_barrier")
if(DPCPP_VERSION STREQUAL "2025.2" OR DPCPP_VERSION STREQUAL "2025.3" OR CUTLASS_SYCL_REVISION STREQUAL "v0.5")
  string(APPEND sycl_link_flags ",+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate")
endif()
string(REPLACE "-fsycl-targets=spir64_gen,spir64" "-fsycl-targets=spir64" sycl_flags "${sycl_flags}")

endif(GPU_LANG STREQUAL "SYCL")
