{
  lib,
  pkgs,
  stdenv,
  torch,
}:

let
  deps = {
    "cutlass_2_10" = [
      pkgs.cutlass_2_10
    ];
    "cutlass_3_5" = [
      pkgs.cutlass_3_5
    ];
    "cutlass_3_6" = [
      pkgs.cutlass_3_6
    ];
    "cutlass_3_8" = [
      pkgs.cutlass_3_8
    ];
    "cutlass_3_9" = [
      pkgs.cutlass_3_9
    ];
    "cutlass_4_0" = [
      pkgs.cutlass_4_0
    ];
    "cutlass_4_4" = [
      pkgs.cutlass_4_4
    ];
    "cutlass_4_5" = [
      pkgs.cutlass_4_5
    ];
    "torch" = [
      torch
    ];
    "sycl_tla" = [
      (torch.xpuPackages.sycl-tla.override { inherit stdenv; })
    ];
    "metal-cpp" = [
      (pkgs.metal-cpp.override { inherit stdenv; }).dev
    ];
  };

  getCppDep = dep: deps.${dep} or (throw "Unknown dependency: ${dep}");
in
deps: lib.flatten (map getCppDep deps)
