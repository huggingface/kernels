{
  lib,
  stdenv,
  buildPythonPackage,
  requireFile,

  autoPatchelfHook,
  proot,
  python,

  # jax is not a hard dependency of the torch_tpu wheel, but TPU kernels
  # import torch_tpu._internal.pallas (jax_op), which needs jax at import
  # time, so it is listed here. jaxlib is not a torch_tpu dependency; it
  # is pulled in transitively by jax (nixpkgs' jax declares it).
  # libtpu is pinned in-tree via pkgs/python-modules/libtpu.
  jax,
  libtpu,

  # The remaining wheel Requires-Dist entries (numpy and absl-py come in
  # transitively through jax).
  frozendict,
  immutabledict,
  portpicker,
  tensorboard,
  torch,
}:

# torch_tpu is served from Google's gated Artifact Registry, which
# requires an OAuth2 bearer token. That token can't be handed to a build
# sandbox without making the derivation impure, so instead the wheel is
# prefetched out-of-band and handed to `requireFile`, which only checks
# that a file matching `name` + `hash` already exists in the store:
#
#   GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token) \
#     nix-builder/scripts/helpers/prefetch-tpu-wheel.sh torch_tpu 0.1.1.dev20260707090224
#
# The script prints the `hash` line to paste in below. The dev build is
# dated; bump the date suffix in `version` below, then re-run the
# script with the new version.

let
  # The wheel ships per-CPython-ABI builds (cp311..cp314); pick the tag
  # matching the python this package set is built for. The hash below is
  # for cp313 (the nixpkgs default python); if either moves, recompute
  # via the prefetch script above.
  abi = "cp${lib.versions.major python.version}${lib.versions.minor python.version}";
in
buildPythonPackage rec {
  pname = "torch-tpu";
  version = "0.1.1.dev20260707090224";
  format = "wheel";

  src = requireFile {
    name = "torch_tpu-${version}-${abi}-${abi}-manylinux_2_31_x86_64.whl";
    hash = "sha256-eCjwoX0UKd/L/IccxXn88p0GdyQjFdPl2Er1luZz4H0="; # cp313
    message = ''
      torch_tpu is served from a gated Google Artifact Registry and
      cannot be fetched by a pure Nix build. Fetch and register it with:

        GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token) \
          nix-builder/scripts/helpers/prefetch-tpu-wheel.sh torch_tpu ${version} ${abi}
    '';
  };

  dependencies = [
    libtpu
    jax
    frozendict
    immutabledict
    portpicker
    tensorboard
    torch
  ];

  # The bundled extensions link against libtorch_python.so / libc10.so /
  # libtorch_cpu.so, which live in torch's site-packages, not on the
  # default search path.
  nativeBuildInputs = [
    autoPatchelfHook
    proot
  ];
  buildInputs = [ stdenv.cc.cc.lib ];

  preFixup = ''
    addAutoPatchelfSearchPath ${torch}/${python.sitePackages}/torch/lib
  '';

  installCheckPhase = ''
    # tcmalloc hard-crashes if /sys is not available:
    # 
    # https://github.com/google/tcmalloc/issues/245
    #
    # We would still like to do an import check with sandboxing enabled,
    # so use proot to fake presence of the relevant part of /sys.
    mkdir -p fake-sys/devices/system/cpu
    echo "0-1" > fake-sys/devices/system/cpu/possible
    PYTHONPATH="$out/${python.sitePackages}:$PYTHONPATH" \
      proot -b fake-sys:/sys \
        ${python.interpreter} -c 'import torch_tpu'
  '';

  meta = with lib; {
    description = "Torch TPU backend (PrivateUse1 name: \"tpu\")";
    homepage = "https://github.com/google-pytorch/torch_tpu";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
