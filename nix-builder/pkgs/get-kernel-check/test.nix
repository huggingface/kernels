{
  lib,
  stdenv,
  python3,
  runCommand,
  writeText,
  get-kernel-check,
  hash-kernel-hook,
  remove-bytecode-hook,
}:

let
  # No accelerator is needed. The Darwin loader setup locates Torch libraries.
  kernels = python3.pkgs.kernels.override { withTorch = false; };
  python = python3.withPackages (ps: lib.optionals stdenv.hostPlatform.isDarwin [ ps.torch ]);
  metadata =
    name: deps:
    writeText "metadata.json" (
      builtins.toJSON {
        inherit name;
        id = "_${builtins.replaceStrings [ "-" ] [ "_" ] name}";
        version = 1;
        license = "Apache-2.0";
        backend.type = "cpu";
        python-depends = [ ];
        kernel-depends = deps;
      }
    );
  dependency = {
    repo-id = "kernels-test/symbols-dependency";
    version = 1;
  };
  dependencyBuild = runCommand "symbols-dependency" { } ''
    mkdir -p $out
    cp ${metadata "symbols-dependency" [ ]} $out/metadata.json
    echo 'VALUE = 42' > $out/__init__.py
  '';
in
stdenv.mkDerivation {
  name = "kernel-symbols-check";
  dontUnpack = true;
  dontConfigure = true;
  dontBuild = true;
  doInstallCheck = true;

  nativeBuildInputs = [
    python
    (get-kernel-check.override {
      inherit kernels;
      python3 = python;
    })
    hash-kernel-hook
    remove-bytecode-hook
  ];

  moduleName = "symbols_test";
  variant = "torch-cpu";
  kernelDeps = writeText "kernel-deps.json" (
    builtins.toJSON [
      {
        inherit dependency;
        path = "${dependencyBuild}";
      }
    ]
  );

  installPhase = ''
    runHook preInstall
    mkdir -p $out/$variant
    cp ${metadata "symbols-test" [ dependency ]} $out/$variant/metadata.json
    chmod u+w $out/$variant/metadata.json
    cp ${./tests/kernel.py} $out/$variant/__init__.py
    cp ${./tests/layers.py} $out/$variant/layers.py
    runHook postInstall
  '';

  # Check the final artifact, after the import check and pre-dist hash hook.
  postPhases = [ "verifySymbolsPhase" ];
  verifySymbolsPhase = ''
    python ${./tests/verify_symbols.py} $out/$variant
  '';
}
