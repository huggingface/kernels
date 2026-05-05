let
  applyOverrides =
    overrides: final: prev:
    prev.lib.mapAttrs (name: value: prev.${name}.overrideAttrs (final.callPackage value { })) overrides;
in
applyOverrides {
  comgr =
    {
      ncurses,
      zlib,
      zstd,
    }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        ncurses
        zlib
        zstd
      ];
    };

  hipblas =
    {
      lib,
      hipblas-common-devel ? null,
    }:
    prevAttrs: {
      propagatedBuildInputs = prevAttrs.buildInputs ++ [ hipblas-common-devel ];
    };

  hipblaslt =
    { hip-runtime-amd }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [ hip-runtime-amd ];
    };

  hipify-clang =
    {
      ncurses,
      zlib,
      zstd,
    }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        ncurses
        zlib
        zstd
      ];
    };

  hiprand =
    { hip-runtime-amd, rocrand }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        hip-runtime-amd
        rocrand
      ];
    };

  openmp-extras-devel =
    { ncurses, zlib }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        ncurses
        zlib
      ];
    };

  openmp-extras-runtime =
    { rocm-llvm }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        rocm-llvm
      ];
      # Can we change rocm-llvm to pick these up?
      installPhase = (prevAttrs.installPhase or "") + ''
        addAutoPatchelfSearchPath ${rocm-llvm}/lib/llvm/lib

        # Requires Python 3.12.
        rm -f $out/lib/llvm/share/gdb/python/ompd/ompdModule.so
      '';
    };

  hipsolver =
    {
      lib,
      suitesparse,
    }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        suitesparse
      ];
    };

  hipsparselt =
    { hip-runtime-amd, roctracer }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        hip-runtime-amd
        roctracer
      ];
    };

  hsa-rocr =
    {
      elfutils,
      libdrm,
      numactl,
    }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        elfutils
        libdrm
        numactl
      ];
    };

  rccl =
    { roctracer }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [ roctracer ];
    };

  rocfft =
    { hip-runtime-amd }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [ hip-runtime-amd ];
    };

  rocm-llvm =
    {
      hsa-rocr,
      libxml2,
      ncurses,
      zlib,
      zstd,
    }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        hsa-rocr
        libxml2
        ncurses
        zlib
        zstd
      ];

      installPhase = (prevAttrs.installPhase or "") + ''
        # Dead symlink(s).
        chmod -R +w $out/lib
        rm -f $out/lib/llvm/bin/flang
      '';
    };

  rocm-gdb =
    {
      expat,
      libxcrypt-legacy,
      ncurses,
      python311,
      python312,
      python313,
      xz,
    }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        expat
        libxcrypt-legacy
        python311
        python312
        python313
        ncurses
        xz
      ];
    };

  rocm-smi-lib =
    { libdrm, valgrind }:
    prevAttrs: {
      postInstall = (prevAttrs.postInstall or "") + ''
        substituteInPlace \
          $out/lib/cmake/rocm_smi/rocm_smiTargets.cmake \
          --replace-fail "/usr/include/libdrm" "${libdrm.dev}/include/libdrm" \
          --replace-fail "/usr/include/valgrind" "${valgrind.dev}/include/valgrind"
      '';

    };

  rocminfo =
    { python3 }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [ python3 ];
    };

  rocprofiler =
    { comgr, hsa-amd-aqlprofile }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        comgr
        hsa-amd-aqlprofile
      ];
    };

  rocprofiler-sdk =
    {
      comgr,
      elfutils,
      hsa-amd-aqlprofile,
      libdrm,
      sqlite,
    }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        comgr
        elfutils
        hsa-amd-aqlprofile
        libdrm
        sqlite
      ];
    };

  rocprofiler-sdk-rocpd =
    {
      comgr,
      elfutils,
      hsa-amd-aqlprofile,
      libdrm,
      sqlite,
    }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        comgr
        elfutils
        hsa-amd-aqlprofile
        libdrm
        sqlite
      ];
    };

  rocrand =
    { hip-runtime-amd }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [ hip-runtime-amd ];
    };

  rocsparse =
    { roctracer }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [ roctracer ];
    };

  roctracer =
    { comgr, hsa-rocr }:
    prevAttr: {
      buildInputs = prevAttr.buildInputs ++ [
        comgr
        hsa-rocr
      ];
    };
}
