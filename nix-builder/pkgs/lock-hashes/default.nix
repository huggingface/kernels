{
  replaceVars,
  runCommand,
}:

let
  lockHashes = replaceVars ./lock_hashes.py {
    download = ../fetch-from-huggingface/download.py;
  };
in
runCommand "lock-hashes" { } ''
  mkdir -p $out/bin
  install -Dm0755 ${lockHashes} $out/bin/lock-hashes
  patchShebangs $out/bin/lock-hashes
''
