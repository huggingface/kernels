# Tests for cuda_archs_intersection and hip_archs_intersection.
#
# Run standalone from the repository root:
#
#   cmake -P kernel-builder/tests/arch-intersection.cmake
#
# Or with an explicit path to utils.cmake (used by the Nix check):
#
#   cmake -DUTILS_CMAKE=.../templates/utils.cmake -P arch-intersection.cmake

if(NOT DEFINED UTILS_CMAKE)
  set(UTILS_CMAKE ${CMAKE_CURRENT_LIST_DIR}/../src/pyproject/templates/utils.cmake)
endif()
include(${UTILS_CMAKE})

function(check FUNC DESCRIPTION SRC TGT EXPECTED)
  cmake_language(CALL ${FUNC} _OUT "${SRC}" "${TGT}")
  if(NOT "${_OUT}" STREQUAL "${EXPECTED}")
    message(STATUS "FAIL: ${DESCRIPTION}\n  src:      ${SRC}\n  tgt:      ${TGT}\n  expected: ${EXPECTED}\n  got:      ${_OUT}")
    set(_FAILURES 1 PARENT_SCOPE)
  else()
    message(STATUS "PASS: ${DESCRIPTION} -> [${_OUT}]")
  endif()
endfunction()

set(_FAILURES 0)

# Pure intersection: archs absent from the toolchain list are dropped, even
# where loose matching would have picked them (10.1 for target 10.3); 11.8
# does not cover target 11.0; 12.0 matches exactly.
check(cuda_archs_intersection "kernel archs filtered by toolchain archs"
  "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;11.8;12.0"
  "7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;11.0;12.0;12.1"
  "7.5;8.0;8.6;8.7;8.9;9.0;10.0;12.0")

# An arch-specific variant replaces its base arch.
check(cuda_archs_intersection "a-variant replaces base"
  "7.5;8.0;8.6;9.0;9.0a"
  "8.0;8.9;9.0"
  "8.0;9.0a")

# Variants sharing a base are all emitted.
check(cuda_archs_intersection "variants sharing a base"
  "10.0a;10.0f"
  "10.0"
  "10.0a;10.0f")

# A variant is dropped when its base arch is not a target.
check(cuda_archs_intersection "variant without base target"
  "9.0a"
  "9.1"
  "")

# +PTX matches on the base arch and keeps the suffix.
check(cuda_archs_intersection "PTX match on base"
  "8.0+PTX"
  "8.0;9.0"
  "8.0+PTX")

# +PTX does not match across versions.
check(cuda_archs_intersection "PTX without base target"
  "8.0+PTX"
  "9.0"
  "")

# Suffixes compose: the variant matches its base and +PTX is re-applied.
check(cuda_archs_intersection "a-variant with PTX"
  "9.0a+PTX"
  "9.0"
  "9.0a+PTX")

# Disjoint sets give an empty result (the call site raises FATAL_ERROR).
check(cuda_archs_intersection "empty intersection"
  "7.0;7.2"
  "7.5;8.0"
  "")

# ROCm: exact matches only.
check(hip_archs_intersection "rocm intersection"
  "gfx900;gfx906;gfx908;gfx90a"
  "gfx906;gfx908;gfx1030"
  "gfx906;gfx908")

if(_FAILURES)
  message(FATAL_ERROR "arch intersection tests failed")
else()
  message(STATUS "All arch intersection tests passed")
endif()
