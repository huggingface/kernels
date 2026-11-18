{ runCommand, python3Minimal }:

runCommand "hip-parallel-driver" { } ''
  mkdir -p $out/bin
  substitute ${./hip-parallel-driver.py} $out/bin/hip-parallel-driver \
    --subst-var-by python3 ${python3Minimal}/bin/python3
  chmod +x $out/bin/hip-parallel-driver
''
