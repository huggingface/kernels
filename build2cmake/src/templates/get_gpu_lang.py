#!/usr/bin/env python3

import sys

try:
    import torch
except ImportError:
    print("Torch is required for configuring a kernel build.", file=sys.stderr)
    sys.exit(1)

if torch.version.cuda is not None:
    print("CUDA")
elif torch.version.hip is not None:
    print("HIP")
elif torch.backends.mps.is_available():
    print("Metal")
elif hasattr(torch.version, "xpu") and torch.version.xpu is not None:
    print("SYCL")
else:
    print("CPU")
