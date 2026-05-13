let
  applyOverrides =
    overrides: final: prev:
    prev.lib.mapAttrs (name: value: prev.${name}.overrideAttrs (final.callPackage value { })) overrides;
in
applyOverrides {
  ca-certificates =
    { }:
    prevAttrs: {
      # Clear dependencies.
      buildInputs = [ ];
    };

  chkconfig =
    { }:
    prevAttrs: {
      postInstall = ''
        # Invalid symlink, but we don't need it anyway.
        rm -f $out/lib/systemd/systemd-sysv-install
      '';
    };

  gcc-toolset-14-gcc =
    {
      gcc-toolset-14-binutils,
      libgcc,
      stdenv,
    }:
    prevAttrs: {
      postInstall = ''
        # We don't care about compiling for 32-bit, so yank files to avoid
        # dealing with broken symlinks.
        rm -rf $out/lib/gcc/x86_64-redhat-linux/14/32

        # Fix binutils symlinks.
        for l in $(find $out/libexec -type l); do
          if [ -f ${gcc-toolset-14-binutils}/bin/$(basename $l) ]; then
            ln -sf ${gcc-toolset-14-binutils}/bin/$(basename $l) $l
          fi
        done

        # Not sure yet if we need this.
        find $out -name 'ld.gold' -type l -delete
      '';
    };

  gcc-toolset-14-gcc-cxx =
    { stdenv }:
    prevAttrs: {
      postInstall = ''
        # We don't care about compiling for 32-bit, so yank files to avoid
        # dealing with broken symlinks.
        rm -rf $out/lib/gcc/x86_64-redhat-linux/14/32
      '';
    };

  gcc-toolset-14-gcc-gfortran =
    { }:
    prevAttrs: {
      postInstall = ''
        # We don't care about compiling for 32-bit, so yank files to avoid
        # dealing with broken symlinks.
        rm -rf $out/lib/gcc/x86_64-redhat-linux/14/32
      '';
    };

  glibc =
    { lib, stdenv }:
    prevAttrs: {
      postInstall =
        let
          ldArch = builtins.replaceStrings [ "_" ] [ "-" ] stdenv.hostPlatform.linuxArch;
        in
        ''
          # Update linker script with Nix paths.
          substituteInPlace $out/lib64/libc.so \
            --replace-fail "/lib64/libc.so.6" "$out/lib/libc.so.6" \
            --replace-fail "/usr/lib64/libc_nonshared.a" "$out/lib/libc_nonshared.a" \
            --replace-fail "/lib64/ld-linux-${ldArch}.so.2" "$out/lib/ld-linux-${ldArch}.so.2"
          substituteInPlace $out/lib64/libm.so \
            --replace-fail "/lib64/libm.so.6" "$out/lib/libm.so.6" \
            --replace-fail "/lib64/libmvec.so.1" "$out/lib/libmvec.so.1" \
            --replace-fail "/usr/lib64/libmvec_nonshared.a" "$out/lib/libmvec_nonshared.a"
        '';
    };

  python3-libs =
    { }:
    prevAttrs: {
      postInstall = ''
        mkdir -p $out/lib
        cp -r $out/lib64/python* $out/lib
        rm -rf $out/lib64/python*
      '';
    };
}
