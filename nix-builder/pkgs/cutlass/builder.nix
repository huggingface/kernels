{
  lib,
  fetchFromGitHub,
  cmake,
  cudaPackages,
  python3,
}:

{
  version,
  hash,
}:

cudaPackages.backendStdenv.mkDerivation rec {
  pname = "cutlass";
  inherit version;

  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = pname;
    rev = "v${version}";
    inherit hash;
  };

  nativeBuildInputs = [
    cmake
  ]
  ++ (with cudaPackages; [
    setupCudaHook
    cuda_nvcc
  ]);

  buildInputs = [ python3 ] ++ (with cudaPackages; [ cuda_cudart ]);

  postPatch = lib.optionalString (lib.versionOlder version "2.11.0") ''
    substituteInPlace CMakeLists.txt \
      --replace-fail "{CMAKE_INSTALL_LIBDIR}/cmake/" "{CMAKE_INSTALL_LIBDIR}/cmake/NvidiaCutlass/"
  '';

  cmakeFlags = [
    (lib.cmakeBool "CUTLASS_ENABLE_GTEST_UNIT_TESTS" false)
    (lib.cmakeBool "CUTLASS_ENABLE_HEADERS_ONLY" true)
    (lib.cmakeBool "CUTLASS_ENABLE_TESTS" false)
  ];

  meta = {
    description = "CUDA Templates for Linear Algebra Subroutines";
    homepage = "https://github.com/NVIDIA/cutlass";
    license = lib.licenses.bsd3;
  };
}
