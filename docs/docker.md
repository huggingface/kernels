# Using kernels in a Docker container

build and run the reference [examples/basic.py](examples/basic.py) in a Docker container with the following commands:

```bash
docker build --platform linux/amd64 -t kernels-reference -f docker/Dockerfile.reference .
docker run --gpus all -it --rm -e HF_TOKEN=$HF_TOKEN kernels-reference
```
