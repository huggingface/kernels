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

After `terraform apply`, get the ready-to-use SSH command from the outputs:

```bash
terraform output -raw ssh_command
```

This prints something like:

```bash
ssh -i ~/.ssh/my-key.pem root@10.90.0.x
```

The instance is reachable via its **private IP** (the subnet does not auto-assign public IPs).

### Waiting for first-boot setup

After SSH-ing in, the NixOS configuration is applied in the background by the `amazon-init` service (downloading packages and running `nixos-rebuild switch`). This takes **10–30 minutes**. Run this **inside the VM** to follow progress:

```bash
journalctl -u amazon-init -f
```

The setup is complete when the service reaches `Finished` state, which you can confirm with:

```bash
systemctl status amazon-init
```

> **Note:** `amazon-init` re-runs on every reboot, so the configuration is re-applied each time the instance restarts.

Once done, reload the shell to pick up the aliases and settings:

```bash
exec bash
```

### Inside the VM

Once the setup is complete, a few useful aliases are available:

```bash
ws          # cd /data/workspace  (1 TiB data volume)
ndc         # nix develop -c $SHELL  (enter the Nix dev shell)
nbd         # nix build -L  (build with full logs)
```

If you need a Nix dev shell **before** `amazon-init` finishes (i.e. before `nix-command` and `flakes` are enabled by the rebuild), pass the features explicitly:

```bash
nix --extra-experimental-features 'nix-command flakes' develop
```

Typical workflow for working on a kernel:

```bash
ws
git clone <your-kernel-repo> && cd <your-kernel-repo>
nix develop   # or: ndc
```

`direnv` is also configured, so if the repo has a `.envrc` the dev shell activates automatically on `cd`.

## Teardown

```bash
terraform destroy
```