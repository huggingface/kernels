{
  stdenv,
  lib,
  buildPythonPackage,
  fetchurl,
  python,

  autoAddDriverRunpath,
  autoPatchelfHook,
  pythonWheelDepsCheckHook,

  intervaltree,
  numactl,
  rdma-core,
  torch,
}:

let
  inherit (torch)
    cudaSupport
    cudaPackages
    rocmSupport
    rocmPackages
    ;

  format = "wheel";
  version = "0.1.1";

  # Upstream publishes one wheel per backend on GitHub (only the CUDA 12
  # wheel is also on PyPI). They are all `cp312-abi3` wheels, so they work
  # with Python >= 3.12. The wheels are built for CUDA 12.8, CUDA 13.0, and
  # ROCm 7.1.
  variants = {
    cu12 = {
      localVersion = "";
      hash = "sha256-OQwTIJGJciBlRuRNebEymI8oGOwH4jr80FlfcYORbOw=";
    };
    cu13 = {
      localVersion = "+cu13";
      hash = "sha256-wlHJ0sLM25Omm+DwQzW5JLSnLcjApqyEusQznzQMzHw=";
    };
    rocm = {
      localVersion = "+rocm";
      hash = "sha256-+f+pTq35ZeWsQbEUBSRysjfSWJ56QG7uw8l+Nu5dqw8=";
    };
  };

  variant =
    if cudaSupport then
      "cu${cudaPackages.cudaMajorVersion}"
    else if rocmSupport then
      "rocm"
    else
      throw "uccl is only available for CUDA and ROCm";

  inherit (variants.${variant} or (throw "No uccl wheel available for variant: ${variant}"))
    localVersion
    hash
    ;

  wheelName = "uccl-${version}${localVersion}-cp312-abi3-manylinux_2_35_x86_64.whl";
in
buildPythonPackage {
  inherit format version;
  pname = "uccl";

  src = fetchurl {
    name = wheelName;
    url = "https://github.com/uccl-project/uccl/releases/download/v${version}/${
      builtins.replaceStrings [ "+" ] [ "%2B" ] wheelName
    }";
    inherit hash;
  };

  nativeBuildInputs = [
    autoPatchelfHook
    pythonWheelDepsCheckHook
  ]
  ++ lib.optionals cudaSupport [
    # After autoPatchelfHook to avoid overriding.
    autoAddDriverRunpath
  ];

  buildInputs = [
    stdenv.cc.cc.lib
    numactl
    rdma-core
  ]
  ++ lib.optionals cudaSupport [
    (lib.getLib cudaPackages.cuda_cudart)
  ]
  ++ lib.optionals rocmSupport [
    rocmPackages.clr
  ];

  autoPatchelfIgnoreMissingDeps = lib.optionals cudaSupport [ "libcuda.so.1" ];

  dependencies = [ intervaltree ];

  # The wheels bundle libnuma (and libcudart for CUDA 13) and have impure
  # RPATHs/RUNPATHs (/usr/local/cuda, /opt/rocm, /usr/lib/x86_64-linux-gnu).
  # Use the libraries from Nix instead and drop all RPATHs, autoPatchelfHook
  # will add the right ones.
  postInstall = ''
    sitePackages=$out/${python.sitePackages}

    for lib in $(find $sitePackages/uccl -name '*.so'); do
      for needed in $(patchelf --print-needed $lib); do
        case $needed in
          libnuma-*)
            patchelf --replace-needed $needed libnuma.so.1 $lib
            ;;
          ${lib.optionalString (variant == "cu13") ''
            libcudart-*)
              patchelf --replace-needed $needed libcudart.so.13 $lib
              ;;
          ''}
        esac
      done
    done
    rm -r $sitePackages/uccl.libs

    find $sitePackages/uccl -name '*.so' -exec patchelf --remove-rpath {} \;

    # libibverbs is loaded using dlopen by soname by the P2P library and
    # extension. Make sure that it can be found.
    patchelf --add-needed libibverbs.so.1 \
      $sitePackages/uccl/p2p.abi3.so \
      $sitePackages/uccl/lib/libuccl_p2p.so
  '';

  # ep.abi3.so links against libtorch/libc10, which are not in a standard
  # library directory.
  preFixup = ''
    addAutoPatchelfSearchPath ${torch}/${python.sitePackages}/torch/lib
  '';

  pythonImportsCheck = [ "uccl" ];

  meta = {
    description = "Efficient communication library for GPUs (collectives, P2P and expert parallelism)";
    homepage = "https://github.com/uccl-project/uccl";
    license = lib.licenses.asl20;
    sourceProvenance = with lib.sourceTypes; [ binaryNativeCode ];
    platforms = [ "x86_64-linux" ];
  };
}
