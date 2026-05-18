{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  ninja,
  setupXpuHook,
  oneapi-torch-dev,
  python3,
  ocloc,
}:

let
  effective-oneapi-torch-dev = oneapi-torch-dev.override { inherit stdenv; };
  dpcppVersion = effective-oneapi-torch-dev.version;
  syclTlaVersions = {
    "2025.3" = {
      version = "0.8";
      hash = "sha256-xXAxIDBesjDDOIa6/YsGznyW+5+NpaO1L96lBuqRzrk=";
    };
  };
  syclTlaVersion =
    syclTlaVersions.${lib.versions.majorMinor dpcppVersion}
    or (throw "Unsupported DPC++ version: ${dpcppVersion}");
in

stdenv.mkDerivation rec {
  pname = "sycl-tla";
  inherit (syclTlaVersion) version;

  src = fetchFromGitHub (
    {
      owner = "intel";
      repo = "sycl-tla";
      inherit (syclTlaVersion) hash;
    }
    // (
      if syclTlaVersion ? rev then
        { inherit (syclTlaVersion) rev; }
      else
        { tag = "v${syclTlaVersion.version}"; }
    )
  );

  nativeBuildInputs = [
    cmake
    effective-oneapi-torch-dev
    ninja
    setupXpuHook
    python3
    ocloc
  ];

  cmakeFlags = [
    "-DCMAKE_C_COMPILER=icx"
    "-DCMAKE_CXX_COMPILER=icpx"
    "-DCUTLASS_ENABLE_SYCL=ON"
    "-DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21,intel_gpu_pvc"
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    "-DCUTLASS_ENABLE_GTEST_UNIT_TESTS=OFF"
    "-DCUTLASS_ENABLE_TESTS=OFF"
    "-DCUTLASS_ENABLE_BENCHMARKS=OFF"
    "-DCUTLASS_ENABLE_HEADERS_ONLY=ON"
  ];

  installPhase = ''
        mkdir -p $out/lib $out/include $out/tools/util/include $out/lib/cmake/SyclTla
        cp -rn $src/include/* $out/include/
        cp -rn $src/tools/util/include/* $out/tools/util/include/
        cat > $out/lib/cmake/SyclTla/SyclTlaConfig.cmake <<EOF
    set(CUTLASS_INCLUDE_DIR  "$out/include")
    set(CUTLASS_TOOLS_UTIL_INCLUDE_DIR "$out/tools/util/include")
    add_compile_definitions(CUTLASS_ENABLE_SYCL)
    add_compile_definitions(DPCPP_SYCL_TARGET=intel_gpu_bmg_g21,intel_gpu_pvc)
    add_compile_definitions(SYCL_INTEL_TARGET=1)
    set(ENV{SYCL_PROGRAM_COMPILE_OPTIONS} "-ze-opt-large-register-file")
    set(ENV{IGC_VISAOptions} "-perfmodel")
    set(ENV{IGC_VectorAliasBBThreshold} "10000")
    set(ENV{IGC_ExtraOCLOptions} "-cl-intel-256-GRF-per-thread")
    EOF
  '';
}
