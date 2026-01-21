{
  lib,
  stdenv,

  makeSetupHook,
  makeWrapper,
  markForXpuRootHook,
  rsync,
  writeShellScriptBin,

  intel-oneapi-dpcpp-cpp,
  intel-oneapi-compiler-dpcpp-cpp-runtime,
  intel-oneapi-compiler-shared,
  intel-oneapi-compiler-shared-runtime,
  intel-oneapi-compiler-shared-common,
  intel-oneapi-compiler-dpcpp-cpp-common,
  intel-oneapi-mkl-classic-include,
  intel-oneapi-mkl-core,
  intel-oneapi-mkl-devel,
  intel-oneapi-mkl-sycl,
  intel-oneapi-mkl-sycl-include,
  intel-oneapi-mkl-sycl-blas,
  intel-oneapi-mkl-sycl-lapack,
  intel-oneapi-mkl-sycl-dft,
  intel-oneapi-mkl-sycl-data-fitting,
  intel-oneapi-mkl-sycl-rng,
  intel-oneapi-mkl-sycl-sparse,
  intel-oneapi-mkl-sycl-stats,
  intel-oneapi-mkl-sycl-vm,
  intel-oneapi-common-vars,
  intel-oneapi-tbb,
  intel-oneapi-openmp,
  intel-pti-dev,
  intel-pti,

}:

let
  # Build only the most essential Intel packages for PyTorch
  essentialIntelPackages = [
    # Core DPC++ compiler package and its dependencies
    intel-oneapi-dpcpp-cpp
    # Compiler runtime and shared components
    intel-oneapi-compiler-dpcpp-cpp-runtime
    intel-oneapi-compiler-shared
    intel-oneapi-compiler-shared-runtime
    intel-oneapi-compiler-shared-common
    intel-oneapi-compiler-dpcpp-cpp-common
    # MKL for math operations - most important for PyTorch
    intel-oneapi-mkl-classic-include
    intel-oneapi-mkl-core
    intel-oneapi-mkl-devel
    intel-oneapi-mkl-sycl
    intel-oneapi-mkl-sycl-include
    intel-oneapi-mkl-sycl-blas
    intel-oneapi-mkl-sycl-lapack
    intel-oneapi-mkl-sycl-dft
    intel-oneapi-mkl-sycl-data-fitting
    intel-oneapi-mkl-sycl-rng
    intel-oneapi-mkl-sycl-sparse
    intel-oneapi-mkl-sycl-stats
    intel-oneapi-mkl-sycl-vm
    # Common infrastructure packages
    intel-oneapi-common-vars
    # TBB for threading
    intel-oneapi-tbb
    # OpenMP
    intel-oneapi-openmp
    # PTI (Profiling and Tracing Interface) - required for PyTorch compilation
    intel-pti-dev
    intel-pti
  ];
in
stdenv.mkDerivation {
  pname = "oneapi-torch-dev";
  version = intel-oneapi-dpcpp-cpp.version;

  dontUnpack = true;

  nativeBuildInputs = [
    makeWrapper
    markForXpuRootHook
    rsync
  ];

  installPhase =
    let
      wrapperArgs = [
        "--add-flags -B${stdenv.cc.libc}/lib"
        "--add-flags -B${placeholder "out"}/lib/crt"
        "--add-flags -B${stdenv.cc.cc}/lib/gcc/${stdenv.hostPlatform.uname.processor}-unknown-linux-gnu/${stdenv.cc.cc.version}"
        "--add-flags '-isysroot ${stdenv.cc.libc_dev}'"
        "--add-flags '-isystem ${stdenv.cc.libc_dev}/include'"
        "--add-flags -I${stdenv.cc.cc}/include/c++/${stdenv.cc.version}"
        "--add-flags -I${stdenv.cc.cc}/include/c++/${stdenv.cc.version}/x86_64-unknown-linux-gnu"
        "--add-flags -L${stdenv.cc.cc}/lib"
        "--add-flags -L${stdenv.cc.cc}/lib/gcc/${stdenv.hostPlatform.uname.processor}-unknown-linux-gnu/${stdenv.cc.version}"
        "--add-flags -L${stdenv.cc.cc.libgcc}/lib"
      ];
    in
    ''
      # Merge all top-level directories from every package into $out using rsync
      for pkg in ${lib.concatStringsSep " " essentialIntelPackages}; do
        rsync -a --exclude=nix-support $pkg/ $out/
      done

      chmod -R u+w $out

      # The `complex` header is only compatible with Intel C++ compilers,
      # but it often ends up in the include paths, causing g++ to fail. So
      # let's just remove it.
      rm $out/include/complex

      wrapProgram $out/bin/icx ${lib.concatStringsSep " " wrapperArgs}
      wrapProgram $out/bin/icpx ${lib.concatStringsSep " " wrapperArgs}
    '';

  dontStrip = true;

  meta = with lib; {
    description = "Intel oneAPI development environment for PyTorch (copied files)";
    longDescription = ''
      A development package for PyTorch compilation with Intel optimizations.
      Uses copied files instead of symlinks to avoid path issues.
    '';
    license = licenses.free;
    platforms = platforms.linux;
    maintainers = [ ];
  };
}
