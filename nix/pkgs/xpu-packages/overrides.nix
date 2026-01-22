let
  applyOverrides =
    overrides: final: prev:
    prev.lib.mapAttrs (name: value: prev.${name}.overrideAttrs (final.callPackage value { })) overrides;
in
applyOverrides {
  intel-oneapi-ccl =
    {
      intel-oneapi-compiler-dpcpp-cpp-runtime,
      intel-oneapi-compiler-shared-runtime,
      intel-oneapi-openmp,
    }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        intel-oneapi-compiler-dpcpp-cpp-runtime
        intel-oneapi-compiler-shared-runtime
        intel-oneapi-openmp
      ];
    };

  intel-oneapi-compiler-dpcpp-cpp-runtime =
    { intel-oneapi-compiler-shared-runtime, intel-oneapi-umf }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        intel-oneapi-compiler-shared-runtime
        intel-oneapi-umf
      ];
    };

  intel-oneapi-compiler-shared-common =
    { intel-oneapi-openmp }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        intel-oneapi-openmp
      ];
    };

  intel-oneapi-dpcpp-cpp =
    { intel-oneapi-openmp }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        intel-oneapi-openmp
      ];
    };

  intel-oneapi-mkl-core =
    {
      intel-oneapi-openmp,
      intel-oneapi-compiler-shared-runtime,
      intel-oneapi-mpi,
    }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        intel-oneapi-openmp
        intel-oneapi-compiler-shared-runtime
        intel-oneapi-mpi
      ];
    };

  intel-oneapi-mpi =
    {
      autoAddDriverRunpath,
      intel-oneapi-openmp,
      libpsm2,
      numactl,
      rdma-core,
      ucx,
      util-linux,
    }:
    prevAttrs: {
      nativeBuildInputs = prevAttrs.nativeBuildInputs ++ [ autoAddDriverRunpath ];
      buildInputs = prevAttrs.buildInputs ++ [
        intel-oneapi-openmp
        libpsm2
        numactl
        rdma-core
        ucx
        util-linux
      ];

      autoPatchelfIgnoreMissingDeps = prevAttrs.autoPatchelfIgnoreMissingDeps ++ [
        "libcuda.so.1"
      ];
    };

  intel-oneapi-openmp =
    {
      elfutils,
      intel-oneapi-compiler-dpcpp-cpp-runtime,
      intel-oneapi-compiler-shared-runtime,
      libffi,
    }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        elfutils
        intel-oneapi-compiler-dpcpp-cpp-runtime
        intel-oneapi-compiler-shared-runtime
        libffi
      ];
    };

  intel-oneapi-compiler-shared-runtime =
    { intel-oneapi-openmp, intel-oneapi-tbb }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        #intel-oneapi-openmp
        intel-oneapi-tbb
      ];

      autoPatchelfIgnoreMissingDeps = prevAttrs.autoPatchelfIgnoreMissingDeps ++ [
        "libonnxruntime.1.12.22.721.so"
      ];
    };

  intel-pti =
    { intel-oneapi-compiler-shared-runtime }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [ intel-oneapi-compiler-shared-runtime ];
      postInstall = (prevAttrs.postInstall or "") + ''
        if [ ! -f "$out/lib/libpti_view.so" ]; then
          versioned_pti_view=$(ls "$out/lib"/libpti_view.so.* 2>/dev/null | head -n1)
          if [ -n "$versioned_pti_view" ]; then
            ln -sf "$(basename "$versioned_pti_view")" "$out/lib/libpti_view.so"
          else
            >&2 echo "Cannot find (versioned) libpti_view.so"
            exit 1
          fi
        fi
      '';
    };
}
