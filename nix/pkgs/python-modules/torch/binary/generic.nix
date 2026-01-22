{
  callPackage,
  config,
  lib,
  stdenv,
  symlinkJoin,
  buildPythonPackage,
  fetchurl,

  cudaSupport ? config.cudaSupport,
  metalSupport ? config.metalSupport,
  rocmSupport ? config.rocmSupport,
  tritonSupport ? (!stdenv.hostPlatform.isDarwin),
  xpuSupport ? (config.xpuSupport or false),

  # Native build inputs
  autoAddDriverRunpath,
  autoPatchelfHook,
  python,
  pythonRelaxWheelDepsHook,
  pythonWheelDepsCheckHook,

  # Build inputs
  cudaPackages,
  rocmPackages,
  xpuPackages,
  zlib,

  # Python dependencies
  cuda-bindings,
  filelock,
  fsspec,
  jinja2,
  networkx,
  numpy,
  pyyaml,
  requests,
  setuptools,
  sympy,
  triton,
  triton-cuda,
  typing-extensions,

  url,
  hash,
  version,

  effectiveStdenv ? if cudaSupport then cudaPackages.backendStdenv else stdenv,
}:
let
  effectiveTriton =
    if cudaSupport then
      triton-cuda
    else if xpuSupport then
      python.pkgs.triton-xpu_2_8
    else
      triton;

  archs = (import ../archs.nix).${lib.versions.majorMinor version};

  supportedTorchCudaCapabilities =
    let
      inherit (archs) capsPerCudaVersion;
      real = capsPerCudaVersion."${lib.versions.majorMinor cudaPackages.cudaMajorMinorVersion}";
      ptx = lib.map (x: "${x}+PTX") real;
    in
    real ++ ptx;
  supportedCudaCapabilities = lib.intersectLists cudaPackages.flags.cudaCapabilities supportedTorchCudaCapabilities;
  inherit (archs) supportedTorchRocmArchs;

  aotritonVersions = with rocmPackages; {
    "2.8" = aotriton_0_10;
    "2.9" = aotriton_0_11;
    "2.10" = aotriton_0_11_1;
  };

  aotriton =
    let
      torchMajorMinor = lib.versions.majorMinor version;
    in
    aotritonVersions.${torchMajorMinor}
      or (throw "aotriton version is not specified Torch ${torchMajorMinor}");

  rocmtoolkit_joined = symlinkJoin {
    name = "rocm-merged";

    paths = with rocmPackages; [
      aotriton
      clr
      comgr
      hipblas
      hipblas-common-devel
      hipblaslt
      hipfft
      hipify-clang
      hiprand
      hipsolver
      hipsparse
      hipsparselt
      hsa-rocr
      miopen-hip
      rccl
      rocblas
      rocm-core
      rocm-device-libs
      rocm-hip-runtime
      rocm-smi-lib
      rocminfo
      rocrand
      rocsolver
      rocsparse
      roctracer
    ];

    postBuild = ''
      # Fix `setuptools` not being found
      rm -rf $out/nix-support

      # Variables that we want to pass through to downstream derivations.
      mkdir -p $out/nix-support
      echo 'export ROCM_PATH="${placeholder "out"}"' >> $out/nix-support/setup-hook
      echo 'export ROCM_SOURCE_DIR="${placeholder "out"}"' >> $out/nix-support/setup-hook
      echo 'export CMAKE_CXX_FLAGS="-I${placeholder "out"}/include -I${placeholder "out"}/include/rocblas"' >> $out/nix-support/setup-hook
    '';
  };

in
buildPythonPackage {
  pname = "torch";
  inherit version;
  format = "wheel";

  stdenv = effectiveStdenv;

  outputs = [
    "out" # output standard python package
    "cxxdev" # propagated deps for the cmake consumers of torch
  ];
  cudaPropagateToOutput = "cxxdev";
  rocmPropagateToOutput = "cxxdev";

  src = fetchurl {
    inherit url hash;
  };

  nativeBuildInputs = [
    pythonRelaxWheelDepsHook
    pythonWheelDepsCheckHook
  ]
  ++ lib.optionals stdenv.hostPlatform.isLinux [
    autoPatchelfHook
  ]
  ++ lib.optionals cudaSupport [
    autoAddDriverRunpath
    cudaPackages.setupCudaHook
  ]
  ++ lib.optionals rocmSupport [
    rocmPackages.setupRocmHook
  ];

  buildInputs =
    lib.optionals cudaSupport (
      with cudaPackages;
      [
        # Use lib output to avoid libcuda.so.1 stub getting used.
        cuda_cudart
        cuda_cupti
        cuda_nvrtc
        cudnn
        libcublas
        libcufft
        libcufile
        libcurand
        libcusolver
        libcusparse
        libcusparse_lt
        nccl
      ]
    )
    ++ lib.optionals (cudaSupport && lib.versionAtLeast version "2.9") [
      cudaPackages.libnvshmem
    ]
    ++ lib.optionals rocmSupport ([
      rocmtoolkit_joined
    ])
    ++ lib.optionals xpuSupport (
      with xpuPackages;
      [
        intel-oneapi-ccl
        intel-oneapi-compiler-dpcpp-cpp-runtime
        intel-oneapi-compiler-shared-runtime
        intel-oneapi-mkl-core
        intel-oneapi-mkl-sycl-blas
        intel-oneapi-mkl-sycl-dft
        intel-oneapi-mkl-sycl-lapack
        intel-oneapi-mpi
        intel-pti
      ]
    )
    # Torch on aarch64-linux vendors libgfortran, which requires zlib.
    ++ lib.optionals (stdenv.hostPlatform.isLinux && stdenv.hostPlatform.isAarch64) [ zlib ];

  dependencies = [
    filelock
    fsspec
    jinja2
    networkx
    numpy
    pyyaml
    requests
    setuptools
    sympy
    typing-extensions
  ]
  ++ lib.optionals tritonSupport [
    effectiveTriton
  ]
  ++ lib.optionals (cudaSupport && lib.versionAtLeast version "2.10") [
    cuda-bindings
  ];

  pythonRelaxWheelDeps = [
    "sympy"
    "triton"
  ];

  # These are framework dependencies that are normally installed as Python
  # dependencies, but we don't need them or provide them because we burn
  # the Nix store paths of the framework into the Torch libraries..
  pythonRemoveWheelDeps =
    lib.optionals cudaSupport [
      "nvidia-cuda-runtime"
      "nvidia-cuda-nvrtc"
      "nvidia-cuda-cupti"
      "nvidia-cudnn"
      "nvidia-cublas"
      "nvidia-cufft"
      "nvidia-curand"
      "nvidia-cusolver"
      "nvidia-cusparse"
      "nvidia-cusparselt"
      "nvidia-nccl"
      "nvidia-nvshmem"
      "nvidia-nvtx"
      "nvidia-nvjitlink"
      "nvidia-cufile"
    ]
    ++ lib.optionals rocmSupport [
      "pytorch-triton-rocm"
    ]
    ++ lib.optionals xpuSupport [
      "intel-cmplr-lib-rt"
      "intel-cmplr-lib-ur"
      "intel-cmplr-lic-rt"
      "intel-sycl-rt"
      "oneccl-devel"
      "oneccl"
      "impi-rt"
      "onemkl-license"
      "onemkl-sycl-blas"
      "onemkl-sycl-dft"
      "onemkl-sycl-lapack"
      "onemkl-sycl-rng"
      "onemkl-sycl-sparse"
      "dpcpp-cpp-rt"
      "intel-opencl-rt"
      "mkl"
      "intel-openmp"
      "tbb"
      "tcmlib"
      "umf"
      "intel-pti"
      "pytorch-triton-xpu"
    ];

  propagatedCxxBuildInputs = lib.optionals rocmSupport [ rocmtoolkit_joined ];

  postInstall =
    lib.optionalString cudaSupport ''
      # Remove to use FindCUDAToolkit from CMake.
      rm -f $out/${python.sitePackages}/torch/share/cmake/Caffe2/FindCUDAToolkit.cmake
    ''
    + lib.optionalString rocmSupport ''
      # Remove all ROCm libraries, we want to link against Nix packages.
      # This keeps the outputs lean and requires downstream to specify
      # dependencies.
      rm -rf $out/${python.sitePackages}/torch/lib/{libamd*,libaotriton*,libdrm*,libelf*,libgomp*,libhip*,libhsa*,libMIOpen*,libnuma*,librccl*,libroc*,libtinfo*}.so*
      rm -rf $out/${python.sitePackages}/torch/lib/{rocblas,hipblaslt,hipsparselt}
    '';

  autoPatchelfIgnoreMissingDeps = lib.optionals stdenv.hostPlatform.isLinux [
    "libcuda.so.1"
  ];

  # We want to have glibc in RPATH as well, because kernel-builder build
  # environments use an older glibc.
  autoPatchelfFlags = [ "--keep-libc" ];

  # See https://github.com/NixOS/nixpkgs/issues/296179
  #
  # This is a quick hack to add `libnvrtc` to the runpath so that torch can find
  # it when it is needed at runtime.
  extraRunpaths = lib.optionals cudaSupport [ "${lib.getLib cudaPackages.cuda_nvrtc}/lib" ];
  postPhases = lib.optionals stdenv.hostPlatform.isLinux [ "postPatchelfPhase" ];
  postPatchelfPhase = ''
    while IFS= read -r -d $'\0' elf ; do
      for extra in $extraRunpaths ; do
        echo patchelf "$elf" --add-rpath "$extra" >&2
        patchelf "$elf" --add-rpath "$extra"
      done
    done < <(
      find "''${!outputLib}" "$out" -type f -iname '*.so' -print0
    )
  '';

  postFixup = ''
    mkdir -p "$cxxdev/nix-support"
    printWords "''${propagatedCxxBuildInputs[@]}" >> "$cxxdev/nix-support/propagated-build-inputs"
  '';

  dontStrip = true;

  pythonImportsCheck = [ "torch" ];

  passthru = {
    inherit
      cudaSupport
      cudaPackages
      rocmSupport
      rocmPackages
      xpuSupport
      xpuPackages
      ;

    cudaCapabilities = if cudaSupport then supportedCudaCapabilities else [ ];
    rocmArchs = if rocmSupport then supportedTorchRocmArchs else [ ];
  }
  // (callPackage ../variant.nix {
    torchVersion = version;
  });

  meta = with lib; {
    description = "PyTorch: Tensors and Dynamic neural networks in Python with strong GPU acceleration";
    homepage = "https://pytorch.org/";
    license = lib.licenses.bsd3;
  };
}
