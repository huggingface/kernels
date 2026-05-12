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
    { gcc-toolset-14-binutils }:
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
    { }:
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
