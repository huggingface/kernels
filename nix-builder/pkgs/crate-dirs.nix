{ lib, sourceFiles }:

with lib.fileset;
toSource {
  root = ../..;
  fileset = unions [
    ../../Cargo.lock
    ../../Cargo.toml
    (fileFilter sourceFiles ../../kernel-abi-check)
    (fileFilter sourceFiles ../../kernel-builder)
    (fileFilter sourceFiles ../../kernels-data)
  ];
}
