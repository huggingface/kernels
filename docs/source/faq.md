# FAQ

## Kernel layers

### Why is the kernelization step needed as a separate step?

In earlier versions of `kernels`, a layer's `forward` method was replaced
by `use_kernel_forward_from_hub` and `replace_kernel_forward_from_hub`.
The new `forward` would dispatch to a kernel based on the device type,
whether a model was training, etc. However, this approach was
fundamentally incompatible with `torch.compile` since it relied
on data-dependent branching.

To avoid branching, we have to make dispatch decisions ahead of time,
which is what the `kernelize` function does.

### Why does kernelization only replace `forward` methods?

There are some other possible approaches. The first is to completely
replace existing layers by kernel layers. However, since this would
permit free-form layer classes, it would be much harder to validate
that layers are fully compatible with the layers that they are
replacing. For instance, they could have completely different member
variables. Besides that, we would also need to hold on to the original
layers, in case we need to revert to the base layers when the model
is `kernelize`d again with different options.

A second approach would be to make an auxiliary layer that wraps the
original layer and the kernel layer and dispatches to the kernel layer.
This wouldn't have the issues of the first approach, because kernel layers
could be similarly strict as they are now, and we would still have access
to the original layers when `kernelize`-ing the model again. However,
this would change the graph structure of the model and would break use
cases where programs access the model internals (e.g.
`model.layers[0].attention.query_weight`) or rely on the graph structure
in other ways.

The approach of `forward`-replacement is the least invasive, because
it preserves the original model graph. It is also reversible, since
even though the `forward` of a layer _instance_ might be replaced,
the corresponding class still has the original `forward`.

## Misc

### How can I disable kernel reporting in the user-agent?

By default, we collect telemetry when a call to `get_kernel()` is made.
This only includes the `kernels` version, `torch` version, and the build
information for the kernel being requested.

You can disable this by setting `export DISABLE_TELEMETRY=yes`.
