# Variant of the arch SDK with a backend
# (`sdk-<backend>-<version>-cpython-<version>-<abi>-<system>`).
{
  lib,
  manylinux_2_28,
  cudaPackages,
  python3,

  variant,

  # Backend that is written to the metadata.
  backend,
}:

let
  stdenv = manylinux_2_28.cudaBackendStdenv;
in
stdenv.mkDerivation {
  pname = "sdk-arch-backend-${variant}";
  version = "0.1.0";

  dontUnpack = true;

  dontFixup = true;

  inherit variant backend;

  nativeBuildInputs = [ cudaPackages.cuda_nvcc ];

  buildInputs = [ cudaPackages.cuda_cudart ];

  buildPhase = ''
    runHook preBuild

    # The CUDA runtime is linked statically, so that the extension does not
    # depend on a CUDA library.
    nvccFlags=(
      --compiler-bindir ${stdenv.cc}/bin/g++
      -Xcompiler -fPIC
      -O2
      -Xcompiler -fvisibility=hidden
      -arch=sm_75
      -I${lib.getDev cudaPackages.cuda_cudart}/include
    )

    nvcc "''${nvccFlags[@]}" -c ${./utils.cu} -I${./.} -o utils.o

    # abi3 extensions do not link against libpython.
    ${stdenv.cc}/bin/g++ -fPIC -fvisibility=hidden -O2 -c ${./binding.cpp} -I${./.} \
      -I${python3}/include/python${python3.pythonVersion} -o binding.o

    nvcc "''${nvccFlags[@]}" -shared -cudart static \
      -L${lib.getOutput "static" cudaPackages.cuda_cudart}/lib \
      binding.o utils.o -o _utils.abi3.so
    strip --strip-unneeded _utils.abi3.so

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/$variant
    cp ${./python/__init__.py} $out/$variant/__init__.py
    cp _utils.abi3.so $out/$variant/

    cat > $out/$variant/metadata.json <<EOF
    {
      "name": "sdk-arch-backend",
      "id": "_sdk_arch_backend_test",
      "version": 1,
      "license": "Apache-2.0",
      "python-depends": [],
      "kernel-depends": [],
      "backend": { "type": "$backend" }
    }
    EOF

    runHook postInstall
  '';
}
