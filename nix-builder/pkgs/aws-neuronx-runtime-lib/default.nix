{
  lib,
  stdenv,
  fetchurl,

  autoPatchelfHook,
  dpkg,
  patchelf-old,

  hwloc,
  libfabric-efa,
}:

stdenv.mkDerivation rec {
  pname = "aws-neuronx-runtime-lib";
  version = "2.35.30.0";
  collectivesVersion = "2.35.43.0";

  srcs = [
    (fetchurl {
      url = "https://apt.repos.neuron.amazonaws.com/pool/main/a/aws-neuronx-runtime-lib/aws-neuronx-runtime-lib_${version}-d2725ccb8_amd64.deb";
      hash = "sha256-XKPyBPnmcuK0AGAprZfPrq7zELEnWEnEL81RnP7anyk=";
    })
    (fetchurl {
      url = "https://apt.repos.neuron.amazonaws.com/pool/main/a/aws-neuronx-collectives/aws-neuronx-collectives_${collectivesVersion}-354ad3064_amd64.deb";
      hash = "sha256-AhxqxGQvsNuMRzO745xUrfrPADKWkGq2NIp2HGR+ahE=";
    })
  ];

  nativeBuildInputs = [
    autoPatchelfHook
    dpkg
    patchelf-old
  ];

  buildInputs = [
    # libnccom-net.so
    hwloc.lib
    libfabric-efa
    stdenv.cc.cc.lib
  ];

  unpackPhase = ''
    runHook preUnpack
    for src in $srcs; do
      dpkg-deb -x "$src" .
    done
    runHook postUnpack
  '';

  sourceRoot = "opt/aws/neuron";

  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    runHook preInstall

    mkdir -p "$out"
    cp -a include lib "$out/"
    install -Dm444 share/doc/aws-neuronx-runtime-lib/LICENSE.txt \
      -t "$out/share/doc/aws-neuronx-runtime-lib"

    runHook postInstall
  '';

  # Add rpaths for dlopen'ed libraries.
  preFixup = ''
    addNrtRunpaths() {
      patchelf --add-rpath '$ORIGIN:${lib.getLib libfabric-efa}/lib' \
        "$out/lib/libnrt.so.${version}"
      patchelf --add-rpath '$ORIGIN' "$out/lib/libnccom.so.${collectivesVersion}"
    }
    postFixupHooks+=(addNrtRunpaths)
  '';

  meta = {
    description = "AWS Neuron runtime (libnrt) and collectives (libnccom) libraries";
    homepage = "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/";
    # AWS Neuron License Agreement, which prohibits redistribution.
    license = lib.licenses.unfree;
    platforms = [ "x86_64-linux" ];
    sourceProvenance = [ lib.sourceTypes.binaryNativeCode ];
  };
}
