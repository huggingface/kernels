{
  stdenv,
  wrapCCWith,
  bintools,
  glibc,
  llvm,
  rsync,
}:

(wrapCCWith rec {
  inherit bintools;

  cc = stdenv.mkDerivation {
    inherit (llvm) version;
    pname = "rocm-llvm-clang";

    nativeBuildInputs = [ rsync ];

    dontUnpack = true;

    installPhase = ''
      runHook preInstall

      mkdir -p $out

      for path in ${llvm}/llvm ${bintools}; do
        rsync -a $path/ $out/
      done
      chmod -R u+w $out

      clang_version=`$out/bin/clang --version | grep -E -o "clang version [0-9]+" | cut -d ' ' -f3`

      # Symlink $out/lib entries into the clang lib directory. However,
      # some directories might already exists. In that case, do not override
      # them but symlink entries within the directory.
      resourceLib=$out/lib/clang/$clang_version/lib
      for entry in $out/lib/*; do
        name=$(basename "$entry")
        if [ -d "$entry" ] && [ -d "$resourceLib/$name" ]; then
          for f in "$entry"/*; do
            ln -s "$f" "$resourceLib/$name/$(basename "$f")"
          done
        else
          ln -s "$entry" "$resourceLib/$name"
        fi
      done

      ln -sf $out/include/* $out/lib/clang/$clang_version/include

      # We need to set the version to signal to clang that we want to
      # include HIP/CUDA compatibility headers.
      chmod -R +w $out/share
      mkdir -p $out/share/hip
      cp ${llvm}/share/hip/version $out/share/hip

      runHook postInstall
    '';

    dontStrip = true;

    passthru = {
      isClang = true;
      isROCm = true;
    };
  };

  gccForLibs = stdenv.cc.cc;

  extraPackages = [
    bintools
    glibc
  ];

  nixSupport.cc-cflags = [
    "-resource-dir=$out/resource-root"
    "-fuse-ld=lld"
    "--rocm-device-lib-path=${llvm}/amdgcn/bitcode"
    "-rtlib=compiler-rt"
    "-unwindlib=libunwind"
    "-Wno-unused-command-line-argument"
  ];

  extraBuildCommands = ''
    clang_version=`${cc}/bin/clang --version | grep -E -o "clang version [0-9]+" | cut -d ' ' -f3`
    mkdir -p $out/resource-root
    ln -s ${cc}/lib/clang/$clang_version/{include,lib} $out/resource-root

    echo "" > $out/nix-support/add-hardening.sh

    # The cc wrapper puts absolute paths to the libstdc++ headers here.
    # However, absolute paths put them before the ROCm wrappers. This
    # cause compilation errors in downstream dependencies because e.g.
    # libstdc++'s new operator cannot handle device code.
    echo "" > $out/nix-support/libcxx-cxxflags

    # GPU compilation uses builtin `lld`
    substituteInPlace $out/bin/{clang,clang++} \
      --replace-fail "-MM) dontLink=1 ;;" "-MM | --cuda-device-only) dontLink=1 ;;''\n--cuda-host-only | --cuda-compile-host-device) dontLink=0 ;;"
  '';
}).overrideAttrs
  (_: {
    # aotriton uses unicode characters and the standard nixpkgs wrapper
    # script cannot deal with it. Also see:
    # https://github.com/NixOS/nixpkgs/pull/226166
    wrapper = ./cc-wrapper.sh;
  })
