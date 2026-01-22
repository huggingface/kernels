find_package(NvidiaCutlass)

if (NOT NvidiaCutlass_FOUND)
  set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

# Set CUTLASS_REVISION manually -- its revision detection doesn't work in this case.
  set(CUTLASS_REVISION "v{{ version }}" CACHE STRING "CUTLASS revision to use")


# Use the specified CUTLASS source directory for compilation if CUTLASS_SRC_DIR is provided
  if (DEFINED ENV{CUTLASS_SRC_DIR})
    set(CUTLASS_SRC_DIR $ENV{CUTLASS_SRC_DIR})
  endif()

  if(CUTLASS_SRC_DIR)
    if(NOT IS_ABSOLUTE CUTLASS_SRC_DIR)
      get_filename_component(CUTLASS_SRC_DIR "${CUTLASS_SRC_DIR}" ABSOLUTE)
    endif()
    message(STATUS "The CUTLASS_SRC_DIR is set, using ${CUTLASS_SRC_DIR} for compilation")
    FetchContent_Declare(cutlass SOURCE_DIR ${CUTLASS_SRC_DIR})
  else()
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/nvidia/cutlass.git
        GIT_TAG ${CUTLASS_REVISION}
        GIT_PROGRESS TRUE

        # Speed up CUTLASS download by retrieving only the specified GIT_TAG instead of the history.
        # Important: If GIT_SHALLOW is enabled then GIT_TAG works only with branch names and tags.
        # So if the GIT_TAG above is updated to a commit hash, GIT_SHALLOW must be set to FALSE
        GIT_SHALLOW TRUE
    )
  endif()
  FetchContent_MakeAvailable(cutlass)

  include_directories(${CUTLASS_INCLUDE_DIR})
else()
  message(STATUS "Using system cutlass with version: ${NvidiaCutlass_VERSION}")
endif(NOT NvidiaCutlass_FOUND)
