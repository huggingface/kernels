# Toolchain

## ABI compatibility

Kernels and kernel extensions typically do not have any explicit external
dependencies, except:

- The CUDA library version that they were compiled against.
- The Torch version that they were compiled against.
- The C and C++ standard libraries (glibc and libstdc++ in Linux).
- The Python library.

Of course, the versions on a user's system can differ from the build
system, so we have to account for these dependencies.

### Python

In the case of Python we use the [limited API](https://docs.python.org/3/c-api/stable.html#limited-c-api).
For the limited API, ABI stability it guaranteed. This excludes the
possibility to use some dependencies like pybind11, but since it
reduces the number of builds drastically, we use it anyway.

### CUDA/Torch

Torch and CUDA only have limited ABI compatibility. Therefore, we
compile extensions for all supported CUDA/Torch combinations.

### glibc/libstdc++

glibc and libstdc++ use symbol versioning. As a result, a binary or
library compiled against an older version of these libraries work
on newer versions (modulo the C++11 ABI change). It is however,
not possible to use only older symbol versions when building against
a newer version of these libraries.

The traditional solution to this problem is to build software in
a container that uses an ancient Linux distribution with old
glibc/libstdc++ versions.

With Nix we can do better --- with some work we can compile for old
versions of these libraries using a recent nixpkgs. There are several
nuances:

- libstdc++ is distributed with gcc, so it has the same library version.
- CUDA's nvcc uses a 'backend stdenv'. This stdenv has the latest
  gcc that is supported by nvcc. It can differ from the default gcc,
  because gcc in nixpkgs is sometimes newer than the version supported
  by CUDA.
- gcc also links a binary against libgcc. libgcc must also be compiled
  against the target glibc, otherwise the resulting extensions will
  still rely on symbols from newer glibc versions.

With that in mind, there are (at least?) three ways to do this:

1. Override glibc and libstdc++ system-wide using an overlay.
2. Override the backend stdenv of CUDA with one that has older
   library versions.
3. Only override glibc and libstdc++ through the stdenv for
   the kernel/extension packages.

(1) is the most principled approach -- it guarantees that all packages
use the same library versions, making it impossible for a newer version
to creep in. Unfortunately, this has many issues, libraries and
derivations from simply don't interoperate well with a package set from 2024.
For instance, the build of older glibc versions hangs with GNU
make >= 4.4 due to some dependency cycle.

Overriding the backend stdenv of CUDA (2) has the issue that some
derivations end up with two versions. E.g. they would build using the
headers of the latest glibc and then try to link against the old glibc.

Finally, (3) seems to work really well. We build everything except
the kernels and extensions using unmodified nixpkgs. Then we tell nvcc
to use our modified stdenv using CMake.

To make this possible, we import the glibc and libstd++ derivations
from an old nixpkgs. We then create an intermediate stdenv to rebuild
gcc/libgcc against the old glibc. Then glibc, libstdc++, and the
rebuilt gcc form make a new stdenv together. We also link libstdc++
statically.
