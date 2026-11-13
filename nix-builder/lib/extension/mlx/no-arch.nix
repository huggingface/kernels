{
  lib,
  pkgs,
  stdenv,

  generate-symbols-hook,
  get-kernel-check,
  hash-kernel-hook,
  kernel-builder,
  kernel-layout-check,
  python3,
  remove-bytecode-hook,

  overrideTorch,
}:

{
  buildConfig,

  # Whether to run get-kernel-check.
  doGetKernelCheck ? true,

  # Whether to check the kernel after build.
  doKernelBuildCheck ? true,

  kernelName,

  # Revision to bake into the ops name.
  rev,

  src,

  # Dependencies on other kernels. Path to a JSON file that maps
  # kernel dependencies to output paths.
  kernelDeps,

  # A stringly-typed list of Python dependencies. Ideally we'd take a
  # list of derivations, but we also need to write the dependencies to
  # the output.
  pythonDeps,

  backendPythonDeps,

  # The variant that the build is expected to produce.
  variant,

  # Git provenance (`{ sha; dirty; }`, or `null`) of the kernel source, recorded
  # in the build metadata.
  kernelProvenance,
}:

# Extra validation - the environment should correspind to the build config.
assert (buildConfig.metal or false) -> stdenv.hostPlatform.isDarwin;

let
  inherit
    (import ../../python-deps.nix {
      inherit
        lib
        python3
        overrideTorch
        ;
    })
    resolvePythonDeps
    resolveBackendPythonDeps
    ;
  dependencies =
    resolvePythonDeps pythonDeps
    ++ resolveBackendPythonDeps buildConfig.backend backendPythonDeps
    ++ [ python3.pkgs.mlx ];
  moduleName = builtins.replaceStrings [ "-" ] [ "_" ] kernelName;
  provenanceFlags = import ../provenance-flags.nix { inherit lib kernelProvenance; };

  buildPython = python3.withPackages (ps: [ ps.setuptools ]);
in

stdenv.mkDerivation (prevAttrs: {
  name = "${kernelName}-mlx-ext";

  inherit doKernelBuildCheck kernelDeps moduleName;

  src = pkgs.runCommand "source" { } ''
    mkdir -p $out
    cp -r --no-preserve=mode ${src}/* $out/
    ${pkgs.kernel-builder}/bin/kernel-builder create-pyproject \
      --unique-id ${rev} ${provenanceFlags} $out
  '';

  framework = "mlx";

  env = {
    inherit variant;
  };

  # Add MLX as a dependency, so that devshells also get it as a build input.
  buildInputs = [ python3.pkgs.mlx ];

  nativeBuildInputs = [
    buildPython
    hash-kernel-hook
    kernel-builder
    kernel-layout-check
    remove-bytecode-hook
  ]
  ++ lib.optionals doGetKernelCheck [
    (get-kernel-check.override {
      python3 = python3.withPackages (_: dependencies);
      kernels = overrideTorch python3.pkgs.kernels;
    })
    (generate-symbols-hook.override {
      python3 = python3.withPackages (_: dependencies);
      kernels = overrideTorch python3.pkgs.kernels;
    })
  ];

  buildPhase = ''
    python3 setup.py build_kernel --backend ${buildConfig.backend}
  '';

  installPhase = ''
    mkdir -p $out
    cp -r build/* $out/
  '';

  doInstallCheck = true;

  passthru = {
    inherit dependencies;
    inherit variant;
    archVariant = variant;
  };
})
