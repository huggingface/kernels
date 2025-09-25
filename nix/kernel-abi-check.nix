{
  buildPythonPackage,
  fetchPypi,
  rustPlatform,
}:

buildPythonPackage rec {
  pname = "kernel-abi-check";
  version = "0.6.2";

  src = fetchPypi {
    inherit version;
    pname = "kernel_abi_check";
    hash = "sha256-goWC7SK79FVNEvkp3bISBwbOqdSrmobANtrWIve9/Ys=";
  };

  cargoDeps = rustPlatform.fetchCargoVendor {
    inherit pname version src sourceRoot;
    hash = "sha256-+1jdbKsDKmG+bf0NEVYMv8t7Meuge1z2cgYfbdB9q8A=";
  };

  sourceRoot = "kernel_abi_check-${version}/bindings/python";

  pyproject = true;

  nativeBuildInputs = with rustPlatform; [ cargoSetupHook maturinBuildHook ];
}
