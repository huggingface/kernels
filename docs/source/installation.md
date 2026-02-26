# Installation

Install the `kernels` package with `pip` (requires `torch>=2.5` and CUDA):

```bash
pip install kernels
```

or with `uv`

```bash
uv pip install kernels
```

or if you want the latest version from the `main` branch:

```bash
pip install "kernels[benchmark] @ git+https://github.com/huggingface/kernels#subdirectory=kernels"
```

# Using kernels in a Docker container

Build and run the reference `examples/basic.py` in a Docker container with the following commands:

```bash
docker build --platform linux/amd64 -t kernels-reference -f docker/Dockerfile.reference .
docker run --gpus all -it --rm -e HF_TOKEN=$HF_TOKEN kernels-reference
```
