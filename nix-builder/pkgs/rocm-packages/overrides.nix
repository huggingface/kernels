let
  applyOverrides =
    overrides: final: prev:
    prev.lib.mapAttrs (name: value: prev.${name}.overrideAttrs (final.callPackage value { })) overrides;
in
applyOverrides {
  amdrocm-hipify =
    {
      amdrocm-runtime,
    }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [ amdrocm-runtime ];
    };

  amdrocm-sysdeps =
    { }:
    prevAttrs: {
      postInstall = (prevAttrs.postInstall or "") + ''
        for f in $out/lib/rocm_sysdeps/lib/*; do
          ln -s "$f" "$out/lib/$(basename "$f")"
        done
      '';
    };

  amdrocm-debugger =
    {
      amdrocm-runtime,
      libxcrypt-legacy,
      python311,
      python312,
      python313,
      python314,
    }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        amdrocm-runtime
        libxcrypt-legacy
        python311
        python312
        python313
        python314
      ];
    };

  amdrocm-blas =
    {
      suitesparse,
      fetchFromGitHub,
      cmake,
    }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [
        (suitesparse.overrideAttrs (old: {
          version = "7.11.0";
          src = fetchFromGitHub {
            owner = "DrTimothyAldenDavis";
            repo = "SuiteSparse";
            rev = "v7.11.0";
            hash = "sha256-8CnN2P/W15GpK0nCNoRQongOrzcz5E8l9SgKksqLxd0=";
          };
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ cmake ];
          preConfigure = null;
          makeFlags = null;
          buildFlags = null;
          cmakeFlags = [ "-DSUITESPARSE_ENABLE_PROJECTS=cholmod" ];
          outputs = [ "out" ];
        }))
      ];
    };

  amdrocm-dnn =
    {
      amdrocm-rand,
    }:
    prevAttrs: {
      buildInputs = prevAttrs.buildInputs ++ [ amdrocm-rand ];
    };
}
