{
  cudaSupport ? torch.cudaSupport,
  rocmSupport ? torch.rocmSupport,
  xpuSupport ? torch.xpuSupport,

  lib,
  pkgs,
  stdenv,

  build2cmake,
  get-kernel-check,
  kernel-layout-check,
  python3,
  remove-bytecode-hook,
  writeText,

  torch,
}:

{
  buildConfig,

  # Whether to run get-kernel-check.
  doGetKernelCheck ? true,

  kernelName,

  # Revision to bake into the ops name.
  rev,

  src,

  # A stringly-typed list of Python dependencies. Ideally we'd take a
  # list of derivations, but we also need to write the dependencies to
  # the output.
  pythonDeps,

  backendPythonDeps,
}:

# Extra validation - the environment should correspind to the build config.
assert (buildConfig ? cudaVersion) -> cudaSupport;
assert (buildConfig ? rocmVersion) -> rocmSupport;
assert (buildConfig ? xpuVersion) -> xpuSupport;
assert (buildConfig.metal or false) -> stdenv.hostPlatform.isDarwin;

let
  inherit (import ../deps.nix { inherit lib pkgs torch; }) resolvePythonDeps resolveBackendPythonDeps;
  dependencies =
    resolvePythonDeps pythonDeps
    ++ resolveBackendPythonDeps buildConfig.backend backendPythonDeps
    ++ [ torch ];
  moduleName = builtins.replaceStrings [ "-" ] [ "_" ] kernelName;
  metalSupport = buildConfig.metal or false;
in

stdenv.mkDerivation (prevAttrs: {
  name = "${kernelName}-torch-ext";

  inherit moduleName;

  src = pkgs.runCommand "source" { } ''
    mkdir -p $out
    cp -r --no-preserve=mode ${src}/* $out/
    ${pkgs.build2cmake}/bin/build2cmake generate-torch \
      --ops-id ${rev} $out/build.toml
  '';

  # Add Torch as a dependency, so that devshells for universal kernels
  # also get torch as a build input.
  buildInputs = [ torch ];

  nativeBuildInputs = [
    build2cmake
    kernel-layout-check
    remove-bytecode-hook
  ]
  ++ lib.optionals doGetKernelCheck [
    (get-kernel-check.override {
      python3 = python3.withPackages (_: dependencies);
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
    variant = torch.noarchVariant;
  };
})
