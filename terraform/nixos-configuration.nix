# References:
#   - https://github.com/huggingface/kernels-community/blob/main/.github/workflows/build-pr.yaml
{
  config,
  pkgs,
  lib,
  ...
}:
{
  imports = [
    # Required for EC2 / AWS support (virtio drivers, cloud-init, EBS, etc.)
    <nixpkgs/nixos/modules/virtualisation/amazon-image.nix>
  ];

  system.stateVersion = "25.11";

  # -------------------------------------------------------------------------
  # Nix daemon — mirrors the CI configuration in build-pr.yaml:
  #   max-jobs = 2 / cores = 12 → here we use the full machine instead
  #   sandbox-fallback = false
  #   experimental-features = nix-command flakes
  #   substituters = huggingface cachix
  # -------------------------------------------------------------------------
  nix.settings = {
    experimental-features = [
      "nix-command"
      "flakes"
    ];
    # Sufficient to cater to heavy kernels.
    max-jobs = 4;
    cores = 16;
    sandbox-fallback = false;

    substituters = [
      "https://cache.nixos.org"
      "https://huggingface.cachix.org"
    ];
    trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      # From cachix-action@v16 in build-pr.yaml
      "huggingface.cachix.org-1:ynTPbLS0W8ofXd9fDjk1KvoFky9K2jhxe6r4nXAkc/o="
    ];

    # Allow the main user to add extra substituters without sudo.
    trusted-users = [
      "root"
      "nixos"
    ];
  };

  # Keep build outputs around so incremental rebuilds stay fast.
  nix.gc = {
    automatic = true;
    dates = "weekly";
    options = "--delete-older-than 30d";
  };

  # -------------------------------------------------------------------------
  # Data volume  — format on first boot, then mount at /data.
  # The 1 TiB gp3 EBS volume is attached as /dev/nvme1n1 on Nitro instances.
  # /data/nix-store is bind-mounted over /nix/store so large builds do not
  # fill the root volume.
  # -------------------------------------------------------------------------
  systemd.services.format-data-volume = {
    description = "Format the EBS data volume on first boot if needed";
    wantedBy = [ "multi-user.target" ];
    before = [ "data.mount" ];
    # Only run if the device exists (attachment can lag by a few seconds).
    unitConfig.ConditionPathExists = "/dev/nvme1n1";
    script = ''
      if ! ${pkgs.util-linux}/bin/blkid /dev/nvme1n1 | grep -q ext4; then
        echo "Formatting /dev/nvme1n1 as ext4..."
        ${pkgs.e2fsprogs}/bin/mkfs.ext4 -L kernels-data /dev/nvme1n1
      fi
    '';
    serviceConfig = {
      Type = "oneshot";
      RemainAfterExit = true;
    };
  };

  fileSystems."/data" = {
    device = "/dev/nvme1n1";
    fsType = "ext4";
    options = [
      "nofail"
      "x-systemd.requires=format-data-volume.service"
    ];
  };

  # Bind /nix/store onto the data volume so builds land on the 1 TiB disk.
  fileSystems."/nix/store" = {
    device = "/data/nix-store";
    fsType = "none";
    options = [
      "bind"
      "nofail"
      "x-systemd.requires=data.mount"
    ];
  };

  # Ensure the bind-mount target exists before mounting.
  systemd.tmpfiles.rules = [
    "d /data/nix-store  0755 root root -"
    "d /data/workspace  0755 nixos nixos -"
    "f /var/lib/systemd/linger/root 0644 root root -" 
  ];

  # -------------------------------------------------------------------------
  # Packages for kernel development
  # -------------------------------------------------------------------------
  environment.systemPackages = with pkgs; [
    # Version control & productivity
    git
    git-lfs
    curl
    wget
    jq
    ripgrep
    htop
    iotop
    btop
    tree
    tmux

    # Nix ecosystem tooling
    cachix # binary cache management
    nix-tree # visualise the Nix store graph
    nix-diff # compare two derivations
    direnv # per-directory .envrc / nix develop auto-activation
    nix-direnv # fast direnv integration for Nix

    # Compression (used by the CI closure export/import steps)
    zstd
    gzip
    xz

    # Misc build utilities
    patchelf
    file
    binutils
  ];

  # -------------------------------------------------------------------------
  # Shell environment
  # -------------------------------------------------------------------------

  # direnv hooks for bash and zsh so `nix develop` shells activate automatically.
  programs.direnv = {
    enable = true;
    nix-direnv.enable = true;
  };

  # Useful shell aliases for kernel dev workflow.
  environment.shellAliases = {
    nbd = "nix build -L"; # build with logs
    nbdt = "nix build -L .#ci-test"; # build the CI test output
    ndc = "nix develop -c $SHELL"; # enter dev shell
    ws = "cd /data/workspace";
    dinit = "echo 'use nix' > .envrc && direnv allow"; # init direnv for a flake dir
  };
  
  # -------------------------------------------------------------------------
  # SSH
  # -------------------------------------------------------------------------
  services.openssh = {
    enable = true;
    settings = {
      PermitRootLogin = "prohibit-password"; # key-only root login
      PasswordAuthentication = false;
      X11Forwarding = false;
    };
  };

  # -------------------------------------------------------------------------
  # Firewall — allow SSH only
  # -------------------------------------------------------------------------
  networking.firewall = {
    enable = true;
    allowedTCPPorts = [ 22 ];
  };
}
