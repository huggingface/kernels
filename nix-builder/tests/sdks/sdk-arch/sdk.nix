# Variant of the arch SDK without a backend (`sdk-cpython-<version>-<abi>-<system>`).
{
  manylinux_2_28,
  python3,

  variant,

  backend,
}:

manylinux_2_28.stdenv.mkDerivation {
  pname = "sdk-arch-${variant}";
  version = "0.1.0";

  dontUnpack = true;

  dontFixup = true;

  inherit variant backend;

  buildPhase = ''
    runHook preBuild

    # abi3 extensions do not link against libpython.
    $CC -shared -fPIC -O2 \
      -I${python3}/include/python${python3.pythonVersion} \
      ${./utils.c} -o _utils.abi3.so
    strip --strip-unneeded _utils.abi3.so

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/$variant
    cp ${./python/__init__.py} $out/$variant/__init__.py
    cp _utils.abi3.so $out/$variant/

    cat > $out/$variant/metadata.json <<EOF
    {
      "name": "sdk-arch",
      "id": "_sdk_arch_test",
      "version": 1,
      "license": "Apache-2.0",
      "python-depends": [],
      "kernel-depends": [],
      "backend": { "type": "$backend" }
    }
    EOF

    runHook postInstall
  '';
}
