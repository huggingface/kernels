variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "nixos_channel" {
  description = "NixOS channel to use for the AMI (e.g. '25.11' or '24.11')"
  type        = string
  default     = "25.11"
}

variable "instance_name" {
  description = "Name tag for the EC2 instance (and related resources)"
  type        = string
  default     = "kernels-dev"
}

variable "instance_type" {
  description = "EC2 instance type — heavy on CPU and RAM"
  type        = string
  # 32 vCPUs, 128 GiB RAM
  default = "m7i.8xlarge"
}

variable "root_volume_size_gb" {
  description = "Size of the root EBS volume in GiB"
  type        = number
  default     = 200
}

variable "data_volume_size_gb" {
  description = "Size of the extra data EBS volume in GiB (Nix store, builds, source trees)"
  type        = number
  default     = 1000
}

variable "data_volume_type" {
  description = "EBS volume type for the data volume"
  type        = string
  default     = "gp3"
}

variable "data_volume_iops" {
  description = "Provisioned IOPS for the data volume (gp3 baseline is 3000)"
  type        = number
  default     = 6000
}

variable "data_volume_throughput" {
  description = "Provisioned throughput in MiB/s for the data volume (gp3 baseline is 125)"
  type        = number
  default     = 400
}

variable "subnet_id" {
  description = "ID of the subnet to launch the instance in."
  type        = string
}

variable "security_group_id" {
  description = "ID of an existing security group to attach to the instance."
  type        = string
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair to attach to the instance."
  type        = string
}

variable "ssh_private_key_path" {
  description = "Local path to the private key corresponding to key_pair_name, used by Terraform to provision the instance."
  type        = string
  default     = "~/.ssh/id_rsa"
}

variable "cachix_auth_token" {
  description = "Cachix auth token for pushing to the huggingface cache (optional)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "tags" {
  description = "Extra tags to apply to all resources"
  type        = map(string)
  default     = {}
}
