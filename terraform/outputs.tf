output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.kernels_dev.id
}

output "public_ip" {
  description = "Public IP address of the instance"
  value       = aws_instance.kernels_dev.public_ip
}

output "public_dns" {
  description = "Public DNS name of the instance"
  value       = aws_instance.kernels_dev.public_dns
}

output "ami_id" {
  description = "NixOS AMI used for the instance"
  value       = data.aws_ami.nixos.id
}

output "ami_name" {
  description = "NixOS AMI name (includes channel and git revision)"
  value       = data.aws_ami.nixos.name
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ${var.ssh_private_key_path} root@${aws_instance.kernels_dev.private_ip}"
}
