{
  lib,
  stdenv,
  fetchurl,
  unzip,
}:

let
  version = "3.11";

  binaries = {
    x86_64-linux = {
      suffix = "linux-amd64.tar.gz";
      hash = "sha256-N+2zu89yL5IaAJlBv1h04uDAkmMibJtKLZgHiMsGKrY=";
    };
    aarch64-linux = {
      suffix = "linux-arm64.tar.gz";
      hash = "sha256-Vu1VZuxB0i7J7gcE5qwLmLoQLpI4Tv1TBhc6ItMUx5o=";
    };
    aarch64-darwin = {
      suffix = "arm64-macOS.zip";
      hash = "sha256-FYBr7flRe/6tcuiP5qZpZjXDaR77tuFSFzRA6cW7ULQ=";
    };
    x86_64-darwin = {
      suffix = "x86_64-macOS.zip";
      hash = "sha256-OxwbV/FgESyCHQLyPZRu3ot/V6bM9GMqJaUS0zSpKR8=";
    };
  };

  binary =
    binaries.${stdenv.hostPlatform.system}
      or (throw "pandoc-bin: unsupported system ${stdenv.hostPlatform.system}");
in
stdenv.mkDerivation {
  pname = "pandoc";
  inherit version;

  src = fetchurl {
    url = "https://github.com/jgm/pandoc/releases/download/${version}/pandoc-${version}-${binary.suffix}";
    inherit (binary) hash;
  };

  nativeBuildInputs = lib.optionals stdenv.hostPlatform.isDarwin [ unzip ];

  dontConfigure = true;
  dontBuild = true;

  dontStrip = true;

  # musl binaries, so do not need to be patched.
  dontPatchELF = true;

  installPhase = ''
    runHook preInstall

    install -Dm755 bin/pandoc -t "$out/bin"
    if [ -d share/man ]; then
      mkdir -p "$out/share"
      cp -r share/man "$out/share/"
    fi

    runHook postInstall
  '';

  meta = {
    description = "Conversion between documentation formats";
    homepage = "https://pandoc.org";
    changelog = "https://github.com/jgm/pandoc/releases/tag/${version}";
    license = lib.licenses.gpl2Plus;
    mainProgram = "pandoc";
    platforms = builtins.attrNames binaries;
    sourceProvenance = [ lib.sourceTypes.binaryNativeCode ];
  };
}
