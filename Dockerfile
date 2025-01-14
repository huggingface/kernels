# syntax=docker/dockerfile:1.4
ARG CUDA_VERSION=12.4.0
ARG UBUNTU_VERSION=22.04
ARG TORCH_VERSION=2.5.0

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

# Initialize uv and create virtual env
RUN uv init --app kernel-test

# Move into the app
WORKDIR /app/kernel-test

# Need to re-declare ARG after FROM for use in RUN
ARG CUDA_VERSION
ARG TORCH_VERSION

# Install PyTorch with the appropriate CUDA version

# NOTE: `markupsafe` must be installed first to avoid a conflict with the torch package. 
# See: https://github.com/astral-sh/uv/issues/9647

RUN CUDA_MAJOR_MINOR=$(echo ${CUDA_VERSION} | cut -d'.' -f1,2) && \
    case ${CUDA_MAJOR_MINOR} in \
    "12.1") CUDA_TAG="cu121" ;; \
    "12.2") CUDA_TAG="cu122" ;; \
    "12.4") CUDA_TAG="cu124" ;; \
    *) CUDA_TAG="" ;; \
    esac && \
    if [ -n "${CUDA_TAG}" ]; then \
    echo "Installing PyTorch ${TORCH_VERSION} with CUDA ${CUDA_TAG}" && \
    uv add markupsafe --default-index "https://pypi.org/simple" && \
    uv add "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"; \
    else \
    echo "Installing PyTorch ${TORCH_VERSION} without CUDA-specific index" && \
    uv add "torch==${TORCH_VERSION}"; \
    fi

# Copy application files
COPY kernels ./kernels/kernels
COPY pyproject.toml ./kernels/pyproject.toml
COPY README.md ./kernels/README.md
COPY examples ./examples

# Install the kernel library
RUN uv pip install -e kernels

# Set default command
CMD ["uv", "run", "examples/basic.py"]