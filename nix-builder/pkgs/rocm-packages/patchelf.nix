{
  stdenv,
  cmake,
  fetchFromGitHub,
  patchelf,
  autoPatchelfHook,
}:

# patchelf from nixpkgs returns errors on ROCm shared libraries ("section
# content is not naturally aligned") while still writing out a file. The
# resulting file's PHDR segment is no longer covered by a LOAD segment,
# causing segfaults on dlopen. See:
#
# https://github.com/ROCm/TheRock/issues/6712
#
# The patchelf 0.19.x fixes this corruption, but also added a stricter
# alignment check that hard-errors on the ROCm libraries. So we use
# patchelf 0.18 with a post-release fix for this issue.
final: prev: {
  patchelf = stdenv.mkDerivation {
    pname = "patchelf";
    version = "0.18.0-d0f70ee";

    src = fetchFromGitHub {
      owner = "NixOS";
      repo = "patchelf";
      # Version pinned by TheRock to fix corruption of dylibs.
      rev = "d0f70eea5397606c486857e0a105e53ec123904a";
      hash = "sha256-CWAlYVtcswIpOZj5Qyxswsy1pAnPUTTUDapJO9ar0Mc=";
    };

    nativeBuildInputs = [ cmake ];
    setupHook = patchelf.setupHook;
  };

  # autoPatchelfHook does not take patchelf as an argument. So we override
  # autoPatchelfHook to propagate our patchelf so that it ends up in PATH
  # before patchelf from stdenv.
  autoPatchelfHook = autoPatchelfHook.overrideAttrs (old: {
    propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [ final.patchelf ];
  });
}
