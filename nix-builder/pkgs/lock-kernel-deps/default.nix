{
  python3,
  replaceVarsWith,
}:

let
  python = python3.withPackages (
    ps: with ps; [
      huggingface-hub
      (kernels.override { withTorch = false; })
    ]
  );
in
replaceVarsWith {
  name = "lock-kernel-deps";
  src = ./lock_kernel_deps.py;
  replacements = {
    inherit python;
    download = ../fetch-from-huggingface/download.py;
  };
  dir = "bin";
  isExecutable = true;
}
