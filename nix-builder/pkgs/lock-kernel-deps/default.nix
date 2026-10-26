{
  replaceVars,
  runCommand,
}:

let
  lockKernelDeps = replaceVars ./lock_kernel_deps.py {
    download = ../fetch-from-huggingface/download.py;
  };
in
runCommand "lock-kernel-deps" { } ''
  mkdir -p $out/bin
  install -Dm0755 ${lockKernelDeps} $out/bin/lock-kernel-deps
  patchShebangs $out/bin/lock-kernel-deps
''
