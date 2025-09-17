# Kernels CLI Reference

## Main Functions

### kernels to-wheel

We strongly recommend downloading kernels from the Hub using the `kernels`
package, since this comes with large [benefits](index.md) over using Python
wheels. That said, some projects may require deployment of kernels as
wheels. The `kernels` utility provides a simple solution to this. You can
convert any Hub kernel into a set of wheels with the `to-wheel` command:

```bash
$ kernels to-wheel drbh/img2grey 1.1.2
☸ img2grey-1.1.2+torch27cu128cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch26cu124cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch26cu126cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch27cu126cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch26cu126cxx98-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch27cu128cxx11-cp39-abi3-manylinux_2_28_aarch64.whl
☸ img2grey-1.1.2+torch26cu126cxx98-cp39-abi3-manylinux_2_28_aarch64.whl
☸ img2grey-1.1.2+torch27cu126cxx11-cp39-abi3-manylinux_2_28_aarch64.whl
☸ img2grey-1.1.2+torch26cu126cxx11-cp39-abi3-manylinux_2_28_aarch64.whl
☸ img2grey-1.1.2+torch26cu118cxx98-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch26cu124cxx98-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch26cu118cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
☸ img2grey-1.1.2+torch27cu118cxx11-cp39-abi3-manylinux_2_28_x86_64.whl
```

### kernels upload

Use `kernels upload <dir_containing_build> --repo_id="hub-username/kernel"` to upload
your kernel builds to the Hub.

**Notes**:

- This will take care of creating a repository on the Hub with the `repo_id` provided.
- If a repo with the `repo_id` already exists and if it contains a `build` with the build variant
  being uploaded, it will attempt to delete the files existing under it.
- Make sure to be authenticated (run `hf auth login` if not) to be able to perform uploads to the Hub.

