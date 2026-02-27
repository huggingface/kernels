# Terraform

Spins up an EC2 instance running NixOS, mimicking the infra we
use to develop and build the kernels ourselves.

## Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/install) ≥ 1.5
- AWS credentials in the environment (`AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, etc.)

## Usage

```bash
# 1. Configure
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars          # uncomment and adjust any overrides

# 2. Deploy
terraform init
terraform apply

# 3. Connect (SSH command is printed in outputs)
terraform output ssh_command
```

To push built Nix paths to the Cachix cache, set `cachix_auth_token` in `terraform.tfvars`.

### Connecting

After `terraform apply`, get the SSH command from the outputs:

```bash
terraform output -raw ssh_command
```

This prints something like:

```
ssh -i nixos-key.pem root@<public-ip>
```

The private key (`nixos-key.pem`) is generated automatically and saved in the `terraform/` directory.

## Teardown

```bash
terraform destroy
```

## Inside the VM

TODO