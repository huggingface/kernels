FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# set environment vars
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:${PATH}"

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

# create python script
RUN echo '#!/usr/bin/env python3\n\
    import torch\n\
    import sys\n\
    \n\
    torch_version = "25"\n\
    cuda_version = "124"\n\
    \n\
    kernel_build_path = "/app/activation/build"\n\
    kernel_path = f"{kernel_build_path}/torch{torch_version}-cxx98-cu{cuda_version}-x86_64-linux/"\n\
    sys.path.append(kernel_path)\n\
    \n\
    import activation\n\
    \n\
    # matrix with diagonal increasing values\n\
    x = torch.arange(1, 10).view(3, 3).float()\n\
    x = x.to("cuda")\n\
    print("Input tensor:")\n\
    print(x)\n\
    out = torch.empty_like(x)\n\
    activation.silu_and_mul(out, x)\n\
    print("\\nOutput tensor:")\n\
    print(out)' > main.py

# make the script executable
RUN chmod +x main.py

# initialize uv and create virtual env
RUN uv init --app kernel-test

# install python dependencies
RUN uv pip install torch==2.5.0 numpy

# clone the kernel repository
RUN git clone https://huggingface.co/kernels-community/activation

# set the nvidia runtime env
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# command to run the script
CMD ["uv", "run", "main.py"]