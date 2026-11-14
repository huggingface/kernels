# Derivation attributes for the `add-provenance` hook of `kernel-builder`, from
# the kernel's atomic git provenance (`{ sha; dirty; }`, or `null` when
# unknown). The Nix sandbox has no `.git`, so the kernel's commit SHA and dirty
# state are passed in explicitly.
{ kernelProvenance }:
{
  doAddProvenance = true;
  kernelSha = if kernelProvenance != null then kernelProvenance.sha else null;
  kernelDirty = kernelProvenance != null && kernelProvenance.dirty;
}
