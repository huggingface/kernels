# Variant of the noarch SDK without a backend (`sdk`).
{
  runCommand,

  variant,

  # Backend that is written to the metadata.
  backend,
}:

runCommand "sdk-noarch-${variant}" { inherit variant backend; } ''
  mkdir -p $out/$variant
  cp ${./python/__init__.py} $out/$variant/__init__.py
  cat > $out/$variant/metadata.json <<EOF
  {
    "name": "sdk-noarch",
    "id": "_sdk_noarch_test",
    "version": 1,
    "license": "Apache-2.0",
    "python-depends": [],
    "kernel-depends": [],
    "backend": { "type": "$backend" }
  }
  EOF
''
