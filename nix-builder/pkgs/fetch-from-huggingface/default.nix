{
  lib,
  cacert,
  python3,
  stdenvNoCC,
}:

let
  # `hf download` only accepts the `model`, `dataset` and `space` repository
  # types, so we use `snapshot_download` (which the CLI wraps) directly.
  python = python3.withPackages (ps: [ ps.huggingface-hub ]);
in

{
  # Repository id, e.g. `kernels-community/relu`.
  repoId,

  # Repository type, e.g. `model`, `dataset` or `kernel`.
  type ? "model",

  # Tag or commit SHA.
  revision,

  # SRI hash of the repository snapshot.
  hash,

  # Derivation name, defaults to `<repoId>-<revision>`.
  name ? null,
}:

stdenvNoCC.mkDerivation {
  name = lib.strings.sanitizeDerivationName (if name != null then name else "${repoId}-${revision}");

  nativeBuildInputs = [ python ];

  dontUnpack = true;
  dontConfigure = true;
  dontBuild = true;
  dontFixup = true;

  installPhase = ''
    runHook preInstall

    # Downloading into a local directory bypasses the hub cache, but hf-xet
    # still needs a writable home for its chunk cache.
    export HOME="$NIX_BUILD_TOP"

    # We use a custom script for now, since hf-cli does not support
    # downloading the kernel repo type yet.
    python ${./download.py} \
      ${lib.escapeShellArg repoId} \
      ${lib.escapeShellArg type} \
      ${lib.escapeShellArg revision} \
      "$out"

    runHook postInstall
  '';

  SSL_CERT_FILE = "${cacert}/etc/ssl/certs/ca-bundle.crt";
  HF_HUB_DISABLE_PROGRESS_BARS = "1";
  HF_HUB_DISABLE_TELEMETRY = "1";

  outputHash = hash;
  outputHashAlgo = null;
  outputHashMode = "recursive";

  impureEnvVars = lib.fetchers.proxyImpureEnvVars ++ [ "HF_TOKEN" ];
}
