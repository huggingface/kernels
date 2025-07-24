# FAQ

## Why is the kernelization step needed?

In earlier versions of `kernels`, a layer's `forward` was replaced by
`use_kernel_forward_from_hub` and `replace_kernel_forward_from_hub`. The
new `forward` would dispatch to a kernel based on the device type,
whether a model was training, etc. However, this approach was
fundamentally incompatible with `torch.compile` since it relied
on data-dependent branching.

To avoid branching, we have to make dispatch decisions ahead of time,
which is what the `kernelize` function does.
