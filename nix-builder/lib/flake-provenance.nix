# Extract atomic git provenance (`{ sha; dirty; }`, or `null`) from a flake's
# `self`. The result is `null` when the flake was built from a non-git source
# (no revision is available, e.g. a local `path:`).
{ lib }:
flakeSelf:
let
  sha =
    if flakeSelf == null then
      null
    else if flakeSelf ? rev then
      flakeSelf.rev
    else if flakeSelf ? dirtyRev then
      lib.removeSuffix "-dirty" flakeSelf.dirtyRev
    else
      null;
in
if sha == null then
  null
else
  {
    inherit sha;
    dirty = !(flakeSelf ? rev);
  }
