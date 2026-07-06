# Extract atomic git provenance (`{ sha; dirty; }`, or `null`) from a flake's
# `self`. The result is `null` when the flake was built from a non-git source
# (no revision is available, e.g. a local `path:`).
{ lib }:
flakeSelf:

if flakeSelf == null then
  null
else if flakeSelf ? rev then
  {
    sha = flakeSelf.rev;
    dirty = false;
  }
else if flakeSelf ? dirtyRev then
  {
    sha = lib.removeSuffix "-dirty" flakeSelf.dirtyRev;
    dirty = true;
  }
else
  null
