# Variant of the noarch SDK for a specific backend (`sdk-<backend>`).
{
  runCommand,

  variant,

  # Unique ID of the build variant.
  sdkId,

  # Backend that is written to the metadata.
  backend,
}:

runCommand "sdk-noarch-backend-${variant}" { inherit variant sdkId backend; } ''
  mkdir -p $out/$variant
  cp ${./python/__init__.py} $out/$variant/__init__.py
  cat > $out/$variant/metadata.json <<EOF
  {
    "name": "sdk-noarch-backend",
    "id": "$sdkId",
    "version": 1,
    "license": "Apache-2.0",
    "python-depends": [],
    "kernel-depends": [],
    "backend": { "type": "$backend" }
  }
  EOF
''
