{
  lib,
  stdenv,

  cctools,
  makeWrapper,
  python3,
}:

stdenv.mkDerivation rec {
  name = "rewrite-nix-paths-macho";

  src = ./rewrite-nix-paths-macho.py;

  dontUnpack = true;

  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin
    local scriptdir="$out/libexec/${name}"
    mkdir -p "$scriptdir"
    cp "$src" "$scriptdir/rewrite-nix-paths-macho.py"

    makeWrapper ${python3}/bin/python3 $out/bin/rewrite-nix-paths-macho \
      --add-flags "$scriptdir/rewrite-nix-paths-macho.py" \
      --prefix PATH : ${lib.makeBinPath [ cctools ]}

    runHook postInstall
  '';

  meta = with lib; {
    description = "Rewrite Nix store paths in macOS binaries to be @rpath-relative";
    platforms = platforms.darwin;
  };
}
