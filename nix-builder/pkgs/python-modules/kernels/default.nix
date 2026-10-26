{
  lib,
  buildPythonPackage,
  rustPlatform,
  kernelsCargoDeps,
  mkKernelsRustSrc,

  huggingface-hub,
  pyyaml,
  sigstore,
  tomlkit,
  torch,

  withTorch ? true,
}:

let
  version =
    (builtins.fromTOML (builtins.readFile ../../../../kernels/pyproject.toml)).project.version;
  cargoFlags = [
    "-m"
    "kernels/Cargo.toml"
  ];
in
buildPythonPackage {
  pname = "kernels";
  inherit version;
  format = "pyproject";

  src = mkKernelsRustSrc {
    crates = [ "kernels" ];
    extraFiles = [
      "kernels/README.md"
      "kernels/pyproject.toml"
    ];
  };

  cargoDeps = kernelsCargoDeps;

  maturinBuildFlags = cargoFlags;

  build-system = [
    rustPlatform.cargoSetupHook
    rustPlatform.maturinBuildHook
  ];

  dependencies = [
    huggingface-hub
    pyyaml
    sigstore
    tomlkit
  ]
  ++ lib.optionals withTorch [
    torch
  ];

  pythonImportsCheck = [
    "kernels"
  ];

  meta = with lib; {
    description = "Python client for the Kernel Hub";
  };
}
