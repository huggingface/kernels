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

  cracklib-dicts =
    { }:
    prevAttrs: {
      postInstall = ''
        cp -r usr/share $out
        ln -sf $out/share/cracklib/pw_dict.hwm $out/lib64/cracklib_dict.hwm
        ln -sf $out/share/cracklib/pw_dict.pwd $out/lib64/cracklib_dict.pwd
        ln -sf $out/share/cracklib/pw_dict.pwi $out/lib64/cracklib_dict.pwi
        rm -rf $out/sbin
      '';
    };

  gcc-toolset-13-gcc =
    {
      gcc-toolset-13-binutils,
      libgcc,
      stdenv,
    }:
    prevAttrs: {
      postInstall = ''
        # We don't care about compiling for 32-bit, so yank files to avoid
        # dealing with broken symlinks.
        rm -rf $out/lib/gcc/x86_64-redhat-linux/13/32

        # Remove binutils symlinks to force gcc to use a wrapped binutils
        # from the environment. Without using a wrapper, no rpaths will
        # be set, etc.
        for l in $(find $out/libexec -type l); do
          if [ -f ${gcc-toolset-13-binutils}/bin/$(basename $l) ]; then
            rm -f $l
          fi
        done

        # Not sure yet if we need this.
        find $out -name 'ld.gold' -type l -delete
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

        # Remove binutils symlinks to force gcc to use a wrapped binutils
        # from the environment. Without using a wrapper, no rpaths will
        # be set, etc.
        for l in $(find $out/libexec -type l); do
          if [ -f ${gcc-toolset-14-binutils}/bin/$(basename $l) ]; then
            rm -f $l
          fi
        done

        # Not sure yet if we need this.
        find $out -name 'ld.gold' -type l -delete
      '';
    };

  gcc-toolset-13-gcc-cxx =
    { stdenv }:
    prevAttrs: {
      postInstall = ''
        # We don't care about compiling for 32-bit, so yank files to avoid
        # dealing with broken symlinks.
        rm -rf $out/lib/gcc/x86_64-redhat-linux/13/32
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
          ldArch = builtins.replaceStrings [ "_" ] [ "-" ] stdenv.hostPlatform.uname.processor;
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

  #shadow-utils = { fakeroot, rpmextract }:
  #prevAttrs: {
  #  unpackPhase = ''
  #    for src in $srcs; do
  #      ${fakeroot}/bin/fakeroot ${rpmextract}/bin/rpmextract "$src"
  #    done
  #  '';
  #};
}
