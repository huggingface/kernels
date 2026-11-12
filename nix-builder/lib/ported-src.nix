# Generate the source of a kernel with `kernel-port`, for kernels that commit
# their recipe in place of the output.
{
  lib,
  fetchgit,
  kernel-port,
  runCommand,
}:

# Kernel directory with `port/port.kdl` and the committed `build.toml` and
# `port-provenance.json`.
path:

let
  # The provenance pins and hashes every source, as `kernels.lock` does for
  # kernel dependencies. A store path has no `.git`, so the commits are passed
  # to `kernel-port`, which fails when they differ from the recipe.
  pins = (lib.importJSON (path + "/port-provenance.json")).sources;
  sources = lib.mapAttrs (
    _: pin:
    fetchgit {
      url = pin.repo;
      rev = pin.commit;
      inherit (pin) hash;
      fetchSubmodules = false;
    }
  ) pins;
  vendorArgs = lib.concatStringsSep " " (
    lib.mapAttrsToList (
      name: pin:
      lib.escapeShellArgs [
        "--vendor"
        "${name}=${sources.${name}}"
        "--vendor-rev"
        "${name}=${pin.commit}"
      ]
    ) (removeAttrs pins [ "upstream" ])
  );

  generated = runCommand "ported-src" { nativeBuildInputs = [ kernel-port ]; } ''
    kernel-port ${path + "/port"}/port.kdl \
      --dir ${sources.upstream} \
      --source-rev ${pins.upstream.commit} \
      ${vendorArgs} \
      --out $out
  '';

  # Keep the same files as `source-set.nix`, which cannot be used here because
  # filesets are evaluated and this source is built. Without this, files such
  # as `py.typed` would end up in the kernel despite `pyext`.
  kernelConfig = import ./kernel-config.nix { inherit lib; } path;
  extConfig =
    kernelConfig.toml.torch or kernelConfig.toml.torch-noarch or kernelConfig.toml.tvm-ffi or { };
  pyExt =
    extConfig.pyext or [
      "py"
      "pyi"
    ];
  listedSrc =
    extConfig.src or [ ]
    ++ lib.flatten (
      lib.mapAttrsToList (
        _: kernel: kernel.src ++ lib.optional (kernel ? cargo-manifest) kernel.cargo-manifest
      ) (kernelConfig.toml.kernel or { })
    );
  keepArgs = lib.escapeShellArgs (
    lib.concatMap (ext: [
      "!"
      "-name"
      "*.${ext}"
    ]) pyExt
    ++ lib.concatMap (src: [
      "!"
      "-path"
      "./${src}"
      "!"
      "-path"
      "./${src}/*"
    ]) listedSrc
  );

  # `build.toml` is read during evaluation, so it is committed. The tree hash in
  # `port-provenance.json` makes a change in the `kernel-port` output visible.
  verified = runCommand "ported-src-verified" { } ''
    status=0
    diff -u ${path + "/build.toml"} ${generated}/build.toml || status=1
    diff -u ${path + "/port-provenance.json"} ${generated}/port-provenance.json || status=1
    if [ "$status" -ne 0 ]; then
      echo "Committed build.toml or port-provenance.json differs from what the recipe generates." >&2
      echo "If the change is expected, update them with:" >&2
      echo "  out=\$(nix build --no-link --print-out-paths .#portedSrc)" >&2
      echo "  install -m 644 \$out/build.toml \$out/port-provenance.json ." >&2
      exit 1
    fi
    cp -r --no-preserve=mode ${generated} $out
    cd $out
    for dir in torch-ext tvm-ffi-ext tests; do
      if [ -d "$dir" ]; then
        find "./$dir" -type f ${keepArgs} -delete
        find "./$dir" -type d -empty -delete
      fi
    done
  '';
in
{
  inherit generated verified;
}
