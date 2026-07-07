{
  lib,

  buildPythonPackage,
  fetchPypi,
  setuptools,

  python,

  level-zero,
}:

buildPythonPackage rec {
  pname = "pyzes";
  version = "0.1.2";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-IWZ/CiOxX8+BwSZ9vIb1qbDVEYKXyuZF5KsxxQ6efWs=";
  };

  build-system = [ setuptools ];

  postInstall = ''
    substituteInPlace $out/${python.sitePackages}/pyzes.py \
      --replace-fail 'libName = "/usr/lib/x86_64-linux-gnu/lib" + libName + ".so.1"' \
                     'libName = "${level-zero}/lib/libze_loader.so.1"'
  '';

  pythonImportsCheck = [ "pyzes" ];

  meta = with lib; {
    description = "Python bindings to the Intel Level-Zero-Driver Library";
    homepage = "https://github.com/oneapi-src/level-zero";
    license = lib.licenses.mit;
    sourceProvenance = with lib.sourceTypes; [ fromSource ];
  };
}
