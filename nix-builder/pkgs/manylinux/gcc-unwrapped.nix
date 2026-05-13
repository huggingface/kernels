final: prev:

let
  mkGccUnwrapped =
    {
      lib,
      stdenvNoCC,

      rsync,

      gcc-toolset-gcc,
      gcc-toolset-gcc-cxx,
      gcc-toolset-libstdcxx-devel,
      glibc-headers,
      kernel-headers,
      libgcc,
      libstdcxx,
    }:

    let
      gccMajor = lib.versions.major gcc-toolset-gcc.version;
    in

    stdenvNoCC.mkDerivation {
      pname = "gcc";
      version = gcc-toolset-gcc.version;

      nativeBuildInputs = [ rsync ];

      dontUnpack = true;

      installPhase = ''
        runHook preInstall

        mkdir $out
        for path in ${gcc-toolset-gcc} ${gcc-toolset-gcc-cxx} ${gcc-toolset-libstdcxx-devel} ${glibc-headers} ${kernel-headers} ${libgcc} ${libstdcxx}; do
          rsync --exclude=nix-support -a $path/ $out/
        done

        chmod -R u+w $out

        # Move around libraries to reflect what Nix expects for gccForLibs.
        mv $out/lib/gcc/${stdenvNoCC.hostPlatform.uname.processor}-redhat-linux/${gccMajor}/{libstdc++*,libgcc_s*,libgomp*} $out/lib

        # Update linker script with Nix paths.
        substituteInPlace $out/lib/libstdc++.so \
          --replace-fail "/usr/lib64/libstdc++.so.6" "$out/lib/libstdc++.so.6"
        substituteInPlace $out/lib/libgcc_s.so \
          --replace-fail "/lib64/libgcc_s.so.1" "$out/lib/libgcc_s.so.1"

        runHook postInstall
      '';
    };
in

builtins.listToAttrs (
  map
    (version: {
      name = "gcc${version}-unwrapped";
      value = final.callPackage mkGccUnwrapped {
        gcc-toolset-gcc = final."gcc-toolset-${version}-gcc";
        gcc-toolset-gcc-cxx = final."gcc-toolset-${version}-gcc-cxx";
        gcc-toolset-libstdcxx-devel = final."gcc-toolset-${version}-libstdcxx-devel";
      };
    })
    [
      "13"
      "14"
    ]
)
