terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.0"
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
}

# ---------------------------------------------------------------------------
# EC2 instance running NixOS
# ---------------------------------------------------------------------------
resource "aws_instance" "kernels_dev" {
  ami           = data.aws_ami.nixos.id
  instance_type = var.instance_type
  key_name                    = var.key_pair_name
  subnet_id                   = var.subnet_id
  associate_public_ip_address = true

  vpc_security_group_ids = [var.security_group_id]

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

# ---------------------------------------------------------------------------
# Apply NixOS configuration
#
# Following the pattern from nix.dev:
#   https://nix.dev/tutorials/nixos/deploying-nixos-using-terraform.html
#
# 1. Wait for SSH to be reachable.
# 2. Upload nixos-configuration.nix to /etc/nixos/configuration.nix.
# 3. Optionally write the Cachix auth token.
# 4. Run nixos-rebuild switch to activate the configuration.
# ---------------------------------------------------------------------------
resource "null_resource" "nixos_config" {
  # Re-apply whenever the configuration file changes.
  triggers = {
    config_sha256 = filesha256("${path.module}/nixos-configuration.nix")
    instance_id   = aws_instance.kernels_dev.id
  }

  connection {
    type        = "ssh"
    host        = aws_instance.kernels_dev.public_ip
    user        = "root"
    private_key = file(pathexpand(var.ssh_private_key_path))
    timeout     = "10m"
  }

  # Upload the NixOS configuration.
  provisioner "file" {
    source      = "${path.module}/nixos-configuration.nix"
    destination = "/etc/nixos/configuration.nix"
  }

  # Write the Cachix auth token if one was provided.
  provisioner "remote-exec" {
    inline = [
      var.cachix_auth_token != "" ? "mkdir -p /root/.config/cachix && printf '{\\n  authToken = \"${var.cachix_auth_token}\";\\n}\\n' > /root/.config/cachix/cachix.dhall && chmod 600 /root/.config/cachix/cachix.dhall" : "true"
    ]
  }

  # Activate the configuration.
  provisioner "remote-exec" {
    inline = [
      "nixos-rebuild switch 2>&1 | tail -20",
    ]
  }

  # Register the huggingface Cachix binary cache — mirrors cachix-action@v16
  # in the upstream build-pr.yaml.  cachix is now available (installed above).
  provisioner "remote-exec" {
    inline = [
      "cachix use huggingface",
    ]
  }

  depends_on = [aws_volume_attachment.data]
}
