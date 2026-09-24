from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="kernel-port-relu-aot",
    version="0.0.0",
    packages=["upstream_relu"],
    install_requires=["torch"],
    ext_modules=[
        CppExtension("upstream_relu._C", ["bindings.cpp", "csrc/relu.cpp"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
