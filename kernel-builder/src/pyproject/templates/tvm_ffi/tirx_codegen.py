"""Generate CUDA C source from a TIRx kernel module.

The input module is a plain Python file that defines TIRx PrimFuncs
(`@T.prim_func` from `tvm.script.tirx`) and exports the ones to build in
`__tirx_kernels__` (a list). Every PrimFunc is compiled through the TIRx
pipeline for the requested architecture and the generated CUDA C for all
kernels is written to the output file. The host-side launchers are not
used; kernels are launched from the extension's tvm-ffi binding, which
declares the generated `extern "C" __global__` kernels.
"""

import argparse
import importlib.util
import sys
from pathlib import Path


def load_module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="TIRx kernel module (.py)")
    parser.add_argument("--output", required=True, help="Generated CUDA source (.cu)")
    parser.add_argument("--arch", required=True, help="Codegen architecture, e.g. sm_89")
    args = parser.parse_args()

    import tvm

    module = load_module(args.input)
    kernels = getattr(module, "__tirx_kernels__", None)
    if not kernels:
        raise SystemExit(
            f"{args.input}: TIRx kernel modules must export `__tirx_kernels__`, "
            "a non-empty list of TIRx PrimFuncs"
        )

    target = tvm.target.Target({"kind": "cuda", "arch": args.arch})
    irmod = tvm.IRModule({f"kernel_{i}": func for i, func in enumerate(kernels)})
    with target:
        exe = tvm.compile(irmod, target=target, tir_pipeline="tirx")

    source = exe.mod.imports[0].inspect_source()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(source)


if __name__ == "__main__":
    main()
