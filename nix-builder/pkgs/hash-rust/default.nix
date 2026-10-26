{
  writeShellApplication,
  python3,
  nix-prefetch-git,
}:

writeShellApplication {
  name = "hash-rust";
  runtimeInputs = [
    python3
    nix-prefetch-git
  ];
  text = ''
    exec python3 ${./hash_rust.py} "$@"
  '';
}
