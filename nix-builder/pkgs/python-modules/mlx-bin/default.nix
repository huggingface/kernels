{
  lib,
  buildPythonPackage,
  fetchPypi,
  python,
  unzip,
}:

let
  version = "0.32.4";

  # The hash below is for cp314, the nixpkgs default Python.
  abi = "cp${lib.versions.major python.version}${lib.versions.minor python.version}";

  platform = "macosx_26_0_arm64";

  mlx-metal = fetchPypi {
    pname = "mlx_metal";
    inherit version platform;
    format = "wheel";
    dist = "py3";
    python = "py3";
    abi = "none";
    hash = "sha256-UU35LUGiholdJWS0sY9NpFCWvXpKDEfhBC80WtWn+TY=";
  };
in
buildPythonPackage {
  pname = "mlx";
  inherit version;
  format = "wheel";

  src = fetchPypi {
    pname = "mlx";
    inherit version platform abi;
    format = "wheel";
    dist = abi;
    python = abi;
    hash = "sha256-HdVkkbRKK/ohf2DaZqXJyImt162dwsvPUmOJ6jHYO+4=";
  };

  nativeBuildInputs = [ unzip ];

  # The extension loads libmlx from `@loader_path/lib`.
  postInstall = ''
    unzip -q ${mlx-metal} -d $out/${python.sitePackages}
  '';

  # mlx-metal is unpacked into this output.
  dontCheckRuntimeDeps = true;

  dontStrip = true;

  pythonImportsCheck = [ "mlx.core" ];

  # Importing `mlx.core` needs access to the Metal device.
  __noChroot = true;

  meta = {
    description = "Array framework for Apple silicon";
    homepage = "https://github.com/ml-explore/mlx";
    license = lib.licenses.mit;
    platforms = [ "aarch64-darwin" ];
    sourceProvenance = [ lib.sourceTypes.binaryNativeCode ];
  };
}
