# Assemble `kernel-builder create-pyproject` flags from the kernel's atomic git
# provenance (`{ sha; dirty; }`, or `null` when unknown). The Nix sandbox has no
# `.git`, so the kernel's commit SHA and dirty state are passed in explicitly.
{
  lib,
  kernelProvenance,
}:
lib.optionalString (kernelProvenance != null) (
  "--kernel-sha ${kernelProvenance.sha}" + lib.optionalString kernelProvenance.dirty " --kernel-dirty"
)
