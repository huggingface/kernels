# Metal kernels 🤘

Instructions on this page assume that you installed Nix with the
[Determinate Nix installer](https://docs.determinate.systems/determinate-nix/).

## Targeted macOS versions

Metal kernels are compiled with `-std=metal3.1` (AIR v26), which requires
macOS 15 or later on ARM64 (Apple Silicon).

| macOS version | Support level |
|---------------|---------------|
| macOS 26+     | Fully supported and tested in CI |
| macOS 15      | Fully supported and tested in CI |
| macOS 14      | Best-effort (builds work, some tests may fail due to MPS memory limits) |

## Requirements

To build a Metal kernel, the following requirements must be met:

- An Xcode installation with the Metal compiler must be available. The build
  system automatically detects the Metal toolchain from available Xcode
  installations.
- On macOS 26+, the Metal Toolchain is a separate download from Xcode:
  `xcodebuild -downloadComponent MetalToolchain`
- On macOS 14/15, Metal ships bundled with Xcode (no separate install needed).
- `xcode-select -p` must point to your Xcode installation, typically
  `/Applications/Xcode.app/Contents/Developer`. If this is not the case,
  you can set the path with:
  `sudo xcode-select -s /path/to/Xcode.app/Contents/Developer`
- The Nix sandbox should be set to `relaxed`, because the Nix derivation
  that builds the kernel must have access to Xcode and the Metal Toolchain.
  You can verify this by checking that `/etc/nix/nix.custom.conf` contains
  the line:

  ```
  sandbox = relaxed
  ```

  If you had to add the line, make sure to restart the Nix daemon:

  ```
  sudo launchctl kickstart -k system/systems.determinate.nix-daemon
  ```

You can check these requirements as follows. First, you can check the Xcode
version as follows:

```bash
$ xcodebuild -version
Xcode 26.1
Build version 17B55
```

On macOS 26+, you can validate that the Metal Toolchain is installed with:

```bash
$ xcodebuild -showComponent metalToolchain
Asset Path: /System/Library/AssetsV2/com_apple_MobileAsset_MetalToolchain/68d8db6212b48d387d071ff7b905df796658e713.asset/AssetData
Build Version: 17B54
Status: installed
Toolchain Identifier: com.apple.dt.toolchain.Metal.32023
Toolchain Search Path: /private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.2.54.0.mDxgz0
```
