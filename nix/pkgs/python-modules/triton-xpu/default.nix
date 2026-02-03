{
  lib,
  stdenv,
  buildPythonPackage,
  fetchFromGitHub,
  cmake,
  pkg-config,
  ninja,
  setuptools,
  wheel,
  distutils,
  libxml2,
  pybind11,
  python,
  triton-llvm,
  xpuPackages,

  torchVersion ? "2.8",
}:

let
  torchTritonVersions = {
    "2.8" = {
      llvm = {
        rev = "e12cbd8339b89563059c2bb2a312579b652560d0";
        hash = "sha256-BtMEDk7P7P5k+s2Eb1Ej2QFvjl0A8gN7Tq/Kiu+Pqe4=";
      };
      triton = {
        rev = "ae324eeac8e102a2b40370e341460f3791353398";
        hash = "sha256-rHy/gIH3pYYlmSd6OuRJdB3mJvzeVI2Iav9rszCeV8I=";
      };
      spirv_llm = {
        rev = "96f5ade29c9088ea89533a7e3ca70a9f9464f343";
        hash = "sha256-phbljvV08uWhFJie7XrodLIc97vU4o7zAI1zonN4krY=";
      };
      spirv_headers = {
        rev = "c9aad99f9276817f18f72a4696239237c83cb775";
        hash = "sha256-/KfUxWDczLQ/0DOiFC4Z66o+gtoF/7vgvAvKyv9Z9OA=";
      };
    };
    "2.9" = {
      llvm = {
        rev = "570885128351868c1308bb22e8ca351d318bc4a1";
        hash = "sha256-zRiE+2u8WxZmee0q1Je7Gch+w/LT+nw/4DV18ZB1sM4=";
      };
      triton = {
        rev = "1b0418a9a454b2b93ab8d71f40e59d2297157fae";
        hash = "sha256-MEpkUl1sFEwT4KPH9nSb+9/CmhM6qbG40EAiZmY4mdU=";
      };
      spirv_llm = {
        rev = "9413a66e04ba34f429b05efe00adff0c1f1e0a58";
        hash = "sha256-sVHIQ6z/G0ZiuUoNEfSeOvC+rD+gd7rmdO+BBCXyCJk=";
      };
      spirv_headers = {
        rev = "9e3836d7d6023843a72ecd3fbf3f09b1b6747a9e";
        hash = "sha256-N8NBAkkpOcbgap4loPJJW6E5bjG+TixCh/HN259RyjI=";
      };
    };
  };
  tritonVersions =
    torchTritonVersions.${torchVersion} or (throw "Unsupported Torch version: ${torchVersion}");

  llvmBase = triton-llvm.override (
    {
      llvmTargetsToBuild = [
        "X86"
        "SPIRV"
      ];
    }
    // lib.optionalAttrs (torchVersion == "2.9") {
      llvmProjectsToBuild = [
        "mlir"
        "llvm"
        "lld"
      ];
    }
  );
  llvm = llvmBase.overrideAttrs (old: {
    src = fetchFromGitHub {
      inherit (tritonVersions.llvm) rev hash;
      owner = "llvm";
      repo = "llvm-project";
    };
    pname = "triton-llvm-xpu";
    outputs = [ "out" ];
  });

  spirvLlvmTranslatorSrc = fetchFromGitHub {
    inherit (tritonVersions.spirv_llm) rev hash;
    owner = "KhronosGroup";
    repo = "SPIRV-LLVM-Translator";
  };

  spirvHeadersSrc = fetchFromGitHub {
    inherit (tritonVersions.spirv_headers) rev hash;
    owner = "KhronosGroup";
    repo = "SPIRV-Headers";
  };

in

buildPythonPackage rec {
  pname = "triton-xpu";
  version = torchVersion;
  pyproject = true;
  dontUseCmakeConfigure = true;

  src = fetchFromGitHub {
    inherit (tritonVersions.triton) rev hash;
    owner = "intel";
    repo = "intel-xpu-backend-for-triton";
  };

  sourceRoot = src.name;

  postPatch = ''
    chmod -R u+w $NIX_BUILD_TOP/source
    ${lib.optionalString (torchVersion == "2.9") ''
      sed -i 's/-Werror//g' $NIX_BUILD_TOP/source/CMakeLists.txt
      sed -i 's/ninja==1.11.1.4/ninja>=1.11.1/' $NIX_BUILD_TOP/source/pyproject.toml
    ''}
    sed -i '/if (NOT SPIRVToLLVMTranslator_FOUND)/,/endif (NOT SPIRVToLLVMTranslator_FOUND)/c\
      set(SPIRVToLLVMTranslator_SOURCE_DIR "${spirvLlvmTranslatorSrc}")\n\
      set(SPIRVToLLVMTranslator_BINARY_DIR \''${CMAKE_CURRENT_BINARY_DIR}/SPIRVToLLVMTranslator-build)\n\
      set(LLVM_CONFIG \''${LLVM_LIBRARY_DIR}/../bin/llvm-config)\n\
      set(LLVM_DIR \''${LLVM_LIBRARY_DIR}/cmake/llvm CACHE PATH "Path to LLVM build dir " FORCE)\n\
      set(LLVM_SPIRV_BUILD_EXTERNAL YES CACHE BOOL "Build SPIRV-LLVM Translator as external" FORCE)\n\
      set(LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR ${spirvHeadersSrc})\n\
      add_subdirectory(\''${SPIRVToLLVMTranslator_SOURCE_DIR} \''${CMAKE_CURRENT_BINARY_DIR}/SPIRVToLLVMTranslator-build)\n\
      set(SPIRVToLLVMTranslator_INCLUDE_DIR \''${SPIRVToLLVMTranslator_SOURCE_DIR}/include CACHE INTERNAL "SPIRVToLLVMTranslator_INCLUDE_DIR")\n\
      find_package_handle_standard_args(\n\
              SPIRVToLLVMTranslator\n\
              FOUND_VAR SPIRVToLLVMTranslator_FOUND\n\
              REQUIRED_VARS\n\
                  SPIRVToLLVMTranslator_SOURCE_DIR)\n\
      ' $NIX_BUILD_TOP/source/third_party/intel/cmake/FindSPIRVToLLVMTranslator.cmake

      substituteInPlace $NIX_BUILD_TOP/source/pyproject.toml \
        --replace-fail 'cmake>=3.20,<4.0' 'cmake>=3.20'
  '';

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
  ];

  build-system = with python.pkgs; [
    lit
    pip
    setuptools
  ];

  buildInputs = [
    llvm
    xpuPackages.oneapi-torch-dev
    pybind11
    libxml2.dev
  ];

  depends = [
    setuptools
    distutils
    wheel
  ];

  # Needd to avoid creating symlink: /homeless-shelter [...]
  preBuild = ''
    export HOME=$(mktemp -d)
  '';

  pythonImportsCheck = [
    "triton"
    "triton.language"
  ];

  # Set LLVM env vars for build
  env = {
    LLVM_INCLUDE_DIRS = "${llvm}/include";
    LLVM_LIBRARY_DIR = "${llvm}/lib";
    LLVM_SYSPATH = "${llvm}";
    TRITON_OFFLINE_BUILD = 1;
    TRITON_BUILD_PROTON = 0;
  };
}
