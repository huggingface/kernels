{
  fetchFromHuggingFace,
  writeText,
}:

# Path to the kernel (flake).
path:

let
  locks =
    if builtins.pathExists "${path}/kernels.lock" then
      builtins.fromJSON (builtins.readFile "${path}/kernels.lock")
    else
      [ ];

  # Map kernel locks to Nix store paths by fetching the kernel.
  paths = map (lock: {
    dependency = lock.dependency;
    path = fetchFromHuggingFace {
      repoId = lock.dependency.repo-id;
      type = "kernel";
      revision = lock.lock.commit;
      inherit (lock.lock) hash;
    };
  }) locks;
in
writeText "kernel-deps" (builtins.toJSON paths)
