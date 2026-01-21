# Metal kernels 🤘

Instructions on this page assume that you installed Nix with the
[Determinate Nix installer](https://docs.determinate.systems/determinate-nix/).

## Targeted macOS versions

Since new macOS versions get [adopted quickly](https://telemetrydeck.com/survey/apple/macOS/versions/),
we only support the latest major macOS version except for the first weeks
after a release, when we also support the previous major version.

We currently support macOS 26.0 and later on ARM64 (Apple silicon).

## Requirements

To build a Metal kernel, the following requirements must be met:

- Xcode 26.x must be available on the build machine.
- `xcode-select -p` must point to the Xcode 26 installation, typically
  `/Applications/Xcode.app/Contents/Developer`. If this is not the case,
  you can set the path with:
  `sudo xcode-select -s /path/to/Xcode.app/Contents/Developer`
- The Metal Toolchain must be installed. Starting with macOS 26, this is
  a separate download from Xcode. You can install it with:
  `xcodebuild -downloadComponent MetalToolchain`
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

The reported version must be 26.0 or newer. Then you can validate that the
Metal Toolchain is installed with:

```bash
$ xcodebuild -showComponent metalToolchain
Asset Path: /System/Library/AssetsV2/com_apple_MobileAsset_MetalToolchain/68d8db6212b48d387d071ff7b905df796658e713.asset/AssetData
Build Version: 17B54
Status: installed
Toolchain Identifier: com.apple.dt.toolchain.Metal.32023
Toolchain Search Path: /private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.2.54.0.mDxgz0
```
