FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# set environment vars
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

# install system deps
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# install git-lfs
RUN git lfs install

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# set working directory
WORKDIR /app

# initialize uv and create virtual env
RUN uv init --app kernel-test

# move into the app
WORKDIR /app/kernel-test

# install python depdencies
RUN uv add torch==2.5.0 numpy

# copy kernels lib
COPY kernels ./kernels/kernels
COPY pyproject.toml ./kernels/pyproject.toml
COPY README.md ./kernels/README.md

# install library
RUN uv pip install -e kernels

# copy examples
COPY examples ./examples

# set the nvidia runtime env
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# command to run the script
CMD ["uv", "run", "examples/basic.py"]
# CMD ["ls", "kernels"]
