{ callPackage }:

let
  builder = callPackage ./generic.nix { };
in
{
  nccl_2_27_5 = builder {
    version = "2.27.5";
    wheels = {
      x86_64-linux.cuda12 = {
        url = "https://files.pythonhosted.org/packages/6e/89/f7a07dc961b60645dbbf42e80f2bc85ade7feb9a491b11a1e973aa00071f/nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl";
        sha256 = "ad730cf15cb5d25fe849c6e6ca9eb5b76db16a80f13f425ac68d8e2e55624457";
      };
      aarch64-linux.cuda12 = {
        url = "https://files.pythonhosted.org/packages/bb/1c/857979db0ef194ca5e21478a0612bcdbbe59458d7694361882279947b349/nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl";
        sha256 = "31432ad4d1fb1004eb0c56203dc9bc2178a1ba69d1d9e02d64a6938ab5e40e7a";
      };
    };
  };

  nccl_2_28_9 = builder {
    version = "2.28.9";
    wheels = {
      x86_64-linux.cuda12 = {
        url = "https://files.pythonhosted.org/packages/4a/4e/44dbb46b3d1b0ec61afda8e84837870f2f9ace33c564317d59b70bc19d3e/nvidia_nccl_cu12-2.28.9-py3-none-manylinux_2_18_x86_64.whl";
        sha256 = "485776daa8447da5da39681af455aa3b2c2586ddcf4af8772495e7c532c7e5ab";
      };
      x86_64-linux.cuda13 = {
        url = "https://files.pythonhosted.org/packages/b0/b4/878fefaad5b2bcc6fcf8d474a25e3e3774bc5133e4b58adff4d0bca238bc/nvidia_nccl_cu13-2.28.9-py3-none-manylinux_2_18_x86_64.whl";
        sha256 = "e4553a30f34195f3fa1da02a6da3d6337d28f2003943aa0a3d247bbc25fefc42";
      };
      aarch64-linux.cuda12 = {
        url = "https://files.pythonhosted.org/packages/08/c4/120d2dfd92dff2c776d68f361ff8705fdea2ca64e20b612fab0fd3f581ac/nvidia_nccl_cu12-2.28.9-py3-none-manylinux_2_18_aarch64.whl";
        sha256 = "50a36e01c4a090b9f9c47d92cec54964de6b9fcb3362d0e19b8ffc6323c21b60";
      };
      aarch64-linux.cuda13 = {
        url = "https://files.pythonhosted.org/packages/39/55/1920646a2e43ffd4fc958536b276197ed740e9e0c54105b4bb3521591fc7/nvidia_nccl_cu13-2.28.9-py3-none-manylinux_2_18_aarch64.whl";
        sha256 = "01c873ba1626b54caa12272ed228dc5b2781545e0ae8ba3f432a8ef1c6d78643";
      };
    };
  };

  nccl_2_29_3 = builder {
    version = "2.29.3";
    wheels = {
      x86_64-linux.cuda12 = {
        url = "https://files.pythonhosted.org/packages/31/5a/cac7d231f322b66caa16fd4b136ebc8e4b18b2805811c2d58dc47210cdea/nvidia_nccl_cu12-2.29.3-py3-none-manylinux_2_18_x86_64.whl";
        sha256 = "35ad42e7d5d722a83c36a3a478e281c20a5646383deaf1b9ed1a9ab7d61bed53";
      };
      aarch64-linux.cuda12 = {
        url = "https://files.pythonhosted.org/packages/28/cf/bcf8bb0c0030b1b9a345331f6281c37d2a8669758521eb93c382f6f87c8f/nvidia_nccl_cu12-2.29.3-py3-none-manylinux_2_18_aarch64.whl";
        sha256 = "6351b79dc7d2cc3d654ea1523616b9eeded71fe9c8da66b71eef9a5d1b2adad4";
      };
    };
  };

  nccl_2_29_7 = builder {
    version = "2.29.7";
    wheels = {
      x86_64-linux.cuda13 = {
        url = "https://files.pythonhosted.org/packages/67/f4/58e4e91b6919367c7aafb8e36fce9aad1a3047e536bf7e2fd560927d3a4c/nvidia_nccl_cu13-2.29.7-py3-none-manylinux_2_18_x86_64.whl";
        sha256 = "edd81538446786ec3b73972543e53bb43bcaf0bfc8ef76cb679fcc390ffe136d";
      };
      aarch64-linux.cuda13 = {
        url = "https://files.pythonhosted.org/packages/72/0d/daf50d44177ee0cbc7ff0a0c91eb5ff676c82be42f9a970bc7597f440c3a/nvidia_nccl_cu13-2.29.7-py3-none-manylinux_2_18_aarch64.whl";
        sha256 = "674a12383e3c38a1bcccae7d4f3633b37852230b6047883cb2f4c2d1b36d9bf5";
      };
    };
  };

  nccl_2_30_7 = builder {
    version = "2.30.7";
    wheels = {
      x86_64-linux.cuda13 = {
        url = "https://files.pythonhosted.org/packages/3f/34/c500f90c7ae641b8e0f98965b36b8a7ac79cc8b296e8d251fe3eb592ee54/nvidia_nccl_cu13-2.30.7-py3-none-manylinux_2_18_x86_64.whl";
        sha256 = "cefa7fdb9710efd0f39c5f1be1d61ff6fc9a996c451265bd7fbdcf9455ed4b50";
      };
      aarch64-linux.cuda13 = {
        url = "https://files.pythonhosted.org/packages/d1/21/a73174c6157101bdf1ffc22b517f76ff0082613989dd9bc8f43e8034caac/nvidia_nccl_cu13-2.30.7-py3-none-manylinux_2_18_aarch64.whl";
        sha256 = "ca786ffa5a647c75d4d1f5cc72a6c4f537947e2ba8823d7c8aaf768e7a7b9f77";
      };
    };
  };
}
