{
  runCommand,
  llvm,
}:

runCommand "rocm-llvm-binutils-${llvm.version}" { preferLocalBuild = true; } ''
  mkdir -p $out/bin

  for prog in ${llvm}/llvm/bin/*; do
    ln -sf $prog $out/bin/$(basename $prog)
  done

  ln -s ${llvm}/llvm/bin/llvm-ar $out/bin/ar
  ln -s ${llvm}/llvm/bin/llvm-as $out/bin/as
  ln -s ${llvm}/llvm/bin/llvm-dwp $out/bin/dwp
  ln -s ${llvm}/llvm/bin/llvm-nm $out/bin/nm
  ln -s ${llvm}/llvm/bin/llvm-objcopy $out/bin/objcopy
  ln -s ${llvm}/llvm/bin/llvm-objdump $out/bin/objdump
  ln -s ${llvm}/llvm/bin/llvm-ranlib $out/bin/ranlib
  ln -s ${llvm}/llvm/bin/llvm-readelf $out/bin/readelf
  ln -s ${llvm}/llvm/bin/llvm-size $out/bin/size
  ln -s ${llvm}/llvm/bin/llvm-strip $out/bin/strip
  ln -s ${llvm}/llvm/bin/lld $out/bin/ld
''
