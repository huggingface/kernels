terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# Official NixOS AMI lookup
# AMIs are published weekly by the NixOS project under AWS account 427812963091.
# See: https://nixos.github.io/amis/
# ---------------------------------------------------------------------------
data "aws_ami" "nixos" {
  most_recent = true
  owners      = ["427812963091"]

  filter {
    name   = "name"
    values = ["nixos/${var.nixos_channel}*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

locals {
  common_tags = merge(var.tags, {
    Project   = "hf-kernels-dev"
    ManagedBy = "terraform"
  })

  # Encode the NixOS configuration as base64 so it can be safely embedded in
  # the user-data script without escaping issues (Nix files contain ${ ... }).
  user_data = base64encode(join("", [
    "#!/bin/sh\n",
    "set -e\n",
    # Wait for the EBS data volume to be attached.
    # Terraform attaches it after instance creation, so it may not be present
    # immediately at boot.  Poll for up to 5 minutes (30 x 10 s).
    "echo 'Waiting for data volume /dev/nvme1n1...'\n",
    "for i in $(seq 1 30); do\n",
    "  [ -b /dev/nvme1n1 ] && break\n",
    "  sleep 10\n",
    "done\n",
    # Format (first boot only) and mount the data volume, then create the
    # directories that NixOS will later bind-mount over /nix/store.
    "if [ -b /dev/nvme1n1 ]; then\n",
    "  if ! blkid /dev/nvme1n1 | grep -q ext4; then\n",
    "    mkfs.ext4 -L kernels-data /dev/nvme1n1\n",
    "  fi\n",
    "  mkdir -p /data\n",
    "  mount /dev/nvme1n1 /data\n",
    "  mkdir -p /data/nix-store /data/workspace\n",
    "fi\n",
    # Decode and write the NixOS configuration.
    "base64 -d > /etc/nixos/configuration.nix << 'B64EOF'\n",
    filebase64("${path.module}/nixos-configuration.nix"),
    "\nB64EOF\n",
    # Write the Cachix auth token if one was provided.
    var.cachix_auth_token != "" ? join("", [
      "mkdir -p /root/.config/cachix\n",
      "printf '{\\n  authToken = \"${var.cachix_auth_token}\";\\n}\\n'",
      " > /root/.config/cachix/cachix.dhall\n",
      "chmod 600 /root/.config/cachix/cachix.dhall\n",
    ]) : "",
    # Apply the configuration (installs all packages including cachix).
    "nixos-rebuild switch 2>&1 | tail -20\n",
    # Register the huggingface Cachix binary cache — mirrors cachix-action@v16.
    "cachix use huggingface\n",
  ]))
}

# ---------------------------------------------------------------------------
# EC2 instance running NixOS
# ---------------------------------------------------------------------------
resource "aws_instance" "kernels_dev" {
  ami           = data.aws_ami.nixos.id
  instance_type = var.instance_type
  key_name      = var.key_pair_name
  subnet_id     = var.subnet_id

  associate_public_ip_address = true

  vpc_security_group_ids = [var.security_group_id]

  # NixOS configuration is applied on first boot via user data.
  # Changing nixos-configuration.nix will replace the instance.
  user_data                   = local.user_data
  user_data_replace_on_change = true

  root_block_device {
    volume_size           = var.root_volume_size_gb
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  metadata_options {
    http_tokens = "required" # IMDSv2
  }

  tags = merge(local.common_tags, { Name = var.instance_name })
}

# ---------------------------------------------------------------------------
# Extra EBS data volume (Nix store spillover, build artefacts, source trees)
# ---------------------------------------------------------------------------
resource "aws_ebs_volume" "data" {
  availability_zone = aws_instance.kernels_dev.availability_zone
  size              = var.data_volume_size_gb
  type              = var.data_volume_type
  iops              = var.data_volume_iops
  throughput        = var.data_volume_throughput
  encrypted         = true

  tags = merge(local.common_tags, { Name = "${var.instance_name}-data" })
}

resource "aws_volume_attachment" "data" {
  device_name  = "/dev/xvdf"
  volume_id    = aws_ebs_volume.data.id
  instance_id  = aws_instance.kernels_dev.id
  force_detach = false
}
