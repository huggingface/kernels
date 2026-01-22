# Kernel Builder Docker Containers

This directory contains two Docker containers for different use cases:

## Root Container (`Dockerfile`)

This container runs as root and can modify file permissions when mounting volumes.

```bash
# Build the container
docker build -t kernel-builder:root -f Dockerfile ..

# Use the container
docker run --mount type=bind,source=$(pwd),target=/kernelcode kernel-builder:root build
```

## User Container (`Dockerfile.user`)

This container runs as a non-root user (nixuser with UID 1000) for more secure environments.

```bash
# Build the container
docker build -t kernel-builder:user -f Dockerfile.user ..

# Important: Prepare a directory with correct permissions
mkdir -p ./build
chown -R 1000:1000 ./build  # Match the UID:GID of nixuser in the container

# Use with proper permissions for the build directory
docker run --mount type=bind,source=$(pwd),target=/home/nixuser/kernelcode \
  --mount type=bind,source=$(pwd)/build,target=/home/nixuser/kernelcode/build \
  kernel-builder:user build
```

## Environment Variables

Both containers support these build options:

```bash
# Set options at build time
docker build -t kernel-builder:custom --build-arg MAX_JOBS=8 --build-arg CORES=2 -f Dockerfile ..

# Or at runtime
docker run -e MAX_JOBS=8 -e CORES=2 --mount type=bind,source=$(pwd),target=/kernelcode kernel-builder:root build
```
