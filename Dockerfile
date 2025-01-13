# syntax=docker/dockerfile:1.4
ARG CUDA_VERSION=12.4.0
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:/root/.cargo/bin:${PATH}" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /app

# Create a stage for each PyTorch version we want to test
FROM base as torch-2.0.0
ARG TORCH_VERSION=2.0.0
RUN uv pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | cut -d'.' -f1,2 | tr -d '.')

FROM base as torch-2.1.0
ARG TORCH_VERSION=2.1.0
RUN uv pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | cut -d'.' -f1,2 | tr -d '.')

FROM base as torch-2.2.0
ARG TORCH_VERSION=2.2.0
RUN uv pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | cut -d'.' -f1,2 | tr -d '.')

FROM base as torch-2.5.0
ARG TORCH_VERSION=2.5.0
RUN uv pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | cut -d'.' -f1,2 | tr -d '.')

# Final stage - choose your PyTorch version
FROM torch-${TORCH_VERSION:-2.5.0}

# Copy application files
COPY kernels ./kernels/kernels
COPY pyproject.toml ./kernels/pyproject.toml
COPY README.md ./kernels/README.md
COPY examples ./examples

# Install the kernel library
RUN uv pip install -e kernels

# Set default command
CMD ["uv", "run", "examples/basic.py"]