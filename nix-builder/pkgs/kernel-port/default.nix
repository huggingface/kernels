{
  mkKernelsRustSrc,
  rustPlatform,
}:

let
  version = (builtins.fromTOML (builtins.readFile ../../../kernel-port/Cargo.toml)).package.version;
  cargoFlags = [
    "-p"
    "kernel-port"
  ];
in
rustPlatform.buildRustPackage {
  inherit version;
  pname = "kernel-port";

  src = mkKernelsRustSrc {
    crates = [ "kernel-port" ];
    # The tests run in the check phase.
    tests = [ "kernel-port" ];
  };

  cargoLock = {
    lockFile = ../../../Cargo.lock;
    outputHashes = {
      "hf-hub-1.1.0" = "sha256-PhF4kLJNnaqoVDNfR1ziafBYPVI7912/H4St0qon0IM=";
    };
  };

  cargoBuildFlags = cargoFlags;
  cargoTestFlags = cargoFlags;

  meta = {
    description = "Port third-party kernels to the Hugging Face Kernels layout";
    mainProgram = "kernel-port";
  };
}
