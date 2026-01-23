FROM nixos/nix:2.30.3

# default build args
ARG MAX_JOBS=4
ARG CORES=4

# Combine RUN commands to reduce layers and improve caching
RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf \
    && echo "max-jobs = $MAX_JOBS" >> /etc/nix/nix.conf \
    && echo "cores = $CORES" >> /etc/nix/nix.conf \
    && nix profile install nixpkgs#cachix nixpkgs#git-lfs nixpkgs#jq \
    && cachix use huggingface
# Copy external files
COPY . /etc/src/
# Set environment variables
ENV MAX_JOBS=${MAX_JOBS}
ENV CORES=${CORES}
# Create directory and setup script
RUN mkdir -p /etc/src && \
    cat <<'EOF' > /etc/src/cli.sh
#!/bin/sh
set -e
# Default values
BUILD_URL=""
DEV_SHELL=0
HELP=0
# CLI usage function
function show_usage {
  echo "Kernel Builder CLI"
  echo ""
  echo "Usage: docker run [docker-options] kernel-builder:dev [command] [options]"
  echo ""
  echo "Commands:"
  echo "  build               Build the kernel extension (default if no command specified)"
  echo "  dev                 Start a development shell"
  echo "  fetch [URL]         Clone and build from a Git URL"
  echo "  help                Show this help message"
  echo ""
  echo "Options:"
  echo "  --jobs, -j NUMBER   Set maximum number of parallel jobs (default: $MAX_JOBS)"
  echo "  --cores, -c NUMBER  Set number of cores per job (default: $CORES)"
  echo ""
  echo "Examples:"
  echo "  docker run --mount type=bind,source=$(pwd),target=/kernelcode kernel-builder:root build"
  echo "  docker run -it --mount type=bind,source=$(pwd),target=/kernelcode kernel-builder:root dev"
  echo "  docker run --mount type=bind,source=$(pwd),target=/kernelcode kernel-builder:root fetch https://huggingface.co/user/repo.git"
}

# Function to generate a basic flake.nix if it doesn't exist
function ensure_flake_exists {
  local work_dir=$1
  if [ ! -f "${work_dir}/flake.nix" ]; then
    echo "No flake.nix found, creating a basic one..."
    cat <<'FLAKE_EOF' > "${work_dir}/flake.nix"
{
  description = "Flake for Torch kernel extension";
  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };
  outputs = { self, kernel-builder, }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
FLAKE_EOF
    echo "flake.nix created. You can customize it as needed."
  else
    echo "flake.nix already exists, skipping creation."
  fi
}
# Function to build the extension
function build_extension {
  local work_dir=$1
  local output_dir=$2
  
  echo "Building Torch Extension Bundle from ${work_dir}"
  cd "${work_dir}"
  
  # Check if work_dir is a git repo and get hash if possible
  if [ -d "${work_dir}/.git" ]; then
    # Mark git as safe to allow commands
    git config --global --add safe.directory "${work_dir}"
    # Try to get git revision
    REV=$(git rev-parse --short=8 HEAD)
    
    # Check if working directory is dirty
    if [ -n "$(git status --porcelain 2)" ]; then
      REV="${REV}-dirty"
    fi
  else
    # Generate random material if not a git repo
    REV=$(dd if=/dev/urandom status=none bs=1 count=10 2>/dev/null | base32 | tr '[:upper:]' '[:lower:]' | head -c 10)
  fi
  echo "Building with rev $REV"
  
  # Check for flake.nix or create one
  ensure_flake_exists "${work_dir}"

  # Make sure the build is up to date
  nix run github:huggingface/kernel-builder#update-build -- build.toml

  # Determine override from flake.nix only
  override=""
  # check if using a local path for kernel-builder
  is_using_path=$(grep -q 'kernel-builder.url *= *"path:' flake.nix && echo "yes" || echo "no")
  if [ "$is_using_path" = "yes" ]; then
    override="--override-input kernel-builder github:huggingface/kernel-builder"
    echo "Detected local path for kernel-builder in flake.nix, overriding to use GitHub source."
  else
    echo "No local path for kernel-builder detected in flake.nix."
  fi

  echo "Building with Nix..."
  nix build . \
    $override \
    --max-jobs $MAX_JOBS \
    --cores $CORES \
    -L
  
  echo "Build completed. Copying results to ${output_dir}"
  mkdir -p "${output_dir}"
  cp -r --dereference ./result/* "${output_dir}/"
  # As root, ensure proper permissions for host access
  chmod -R 777 "${output_dir}"
  echo "Done - results available in ${output_dir}"
}
# Function to start a dev shell
function start_dev_shell {
  echo "Starting development shell..."
  # Check for flake.nix or create one
  ensure_flake_exists "/kernelcode"
  cd /kernelcode
  /root/.nix-profile/bin/nix develop
}
# Function to fetch and build from URL
function fetch_and_build {
  if [ -z "$1" ]; then
    echo "Error: URL required for fetch command"
    show_usage
    exit 1
  fi
  
  local repo_url="$1"
  local src_dir="/tmp/kernel-src"
  local output_dir="/kernelcode/result"
  
  echo "Fetching code from ${repo_url} to ${src_dir}"
  # Create a temporary directory for the clone
  mkdir -p "${src_dir}"
  
  # Clone the repository to the temporary directory
  git lfs install
  git clone "${repo_url}" "${src_dir}"
  
  # Build from the temporary directory and copy results to mounted output
  build_extension "${src_dir}" "${output_dir}"
}
# Parse arguments
COMMAND="build"  # Default command
ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    build|dev|fetch|help)
      COMMAND="$1"
      shift
      ;;
    --jobs|-j)
      MAX_JOBS="$2"
      shift 2
      ;;
    --cores|-c)
      CORES="$2"
      shift 2
      ;;
    -*)
      echo "Unknown option: $1"
      show_usage
      exit 1
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done
# Execute the command
case $COMMAND in
  build)

    # First check if the working directory was set by the cli
    # if it is == "/" then we assume the user did not explictly set it
    if [ "$(pwd)" = "/" ]; then
      # set the working dir as /etc/src
      cd /etc/src

      echo "No working directory set, defaulting to /etc/src"
    fi

    # In the case only a volume is mounted, use that as the source, and in the 
    # case a working directory is set we should be in the correct directory by now
    # get the current directory as the source directory
    SRC_DIR=$(pwd)

    # Add a /build to the source directory
    BUILD_DIR="${SRC_DIR}/build"

    # Ensure the build directory exists
    mkdir -p "${BUILD_DIR}"

    # Log the directories being used to user
    echo "Building from source directory: ${SRC_DIR}"
    echo "Output will be placed in: ${BUILD_DIR}"

    # dump the top of the ls
    ls -la "${SRC_DIR}" | head -20

    # When building existing code, use the mounted directory
    build_extension "${SRC_DIR}" "${BUILD_DIR}"
    ;;
  dev)
    start_dev_shell
    ;;
  fetch)
    fetch_and_build "${ARGS[0]}"
    ;;
  help)
    show_usage
    ;;
  *)
    echo "Unknown command: $COMMAND"
    show_usage
    exit 1
    ;;
esac
EOF
RUN chmod +x /etc/src/cli.sh
# # Create output directory structure
# RUN mkdir -p /kernelcode/build
# # Set up volume for kernelcode
# VOLUME /kernelcode

ENTRYPOINT ["/etc/src/cli.sh"]
