{
  stdenv,
  fetchFromGitHub,
  cmake,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "nvtx";
  version = "3.2.1";

  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "NVTX";
    rev = "v${finalAttrs.version}";
    hash = "sha256-MXluy/I5+SaRx2aF64qF4XZ+u67ERAB9TftbOvYt4GE=";
  };

  nativeBuildInputs = [ cmake ];
})
