# Kernels CLI Reference

## Main Functions

### kernels check

You can use `kernels check` to test compliance of a kernel on the Hub.
This currently checks that the kernel:

- Supports the currently-required Python ABI version.
- Works on supported operating system versions.

For example:

```bash
$ kernels check kernels-community/flash-attn3
Checking variant: torch28-cxx11-cu128-aarch64-linux
  🐍 Python ABI 3.9 compatible
  🐧 manylinux_2_28 compatible
[...]
```

### kernels upload

Use `kernels upload <dir_containing_build> --repo_id="hub-username/kernel"` to upload
your kernel builds to the Hub. To know the supported arguments run: `kernels upload -h`.

**Notes**:

- This will take care of creating a repository on the Hub with the `repo_id` provided.
- If a repo with the `repo_id` already exists and if it contains a `build` with the build variant
  being uploaded, it will attempt to delete the files existing under it.
- Make sure to be authenticated (run `hf auth login` if not) to be able to perform uploads to the Hub.

### kernels create-and-upload-card

Use `kernels create-and-upload-card <kernel_source_dir> --card-path README.md` to generate a basic homepage
for the kernel. Find an example [here](https://hf.co/kernels-community/kernel-card-template). You can
optionally push it to the Hub by specifying a `--repo-id`.
