# Variant of the noarch SDK for a specific backend (`sdk-<backend>`).
{
  runCommand,

  variant,

  # Backend that is written to the metadata.
  backend,
}:

runCommand "sdk-noarch-backend-${variant}" { inherit variant backend; } ''
  mkdir -p $out/$variant
  cp ${./python/__init__.py} $out/$variant/__init__.py
  cat > $out/$variant/metadata.json <<EOF
  {
    "name": "sdk-noarch-backend",
    "id": "_sdk_noarch_backend_test",
    "version": 1,
    "license": "Apache-2.0",
    "python-depends": [],
    "kernel-depends": [],
    "backend": { "type": "$backend" }
  }
  EOF
''
