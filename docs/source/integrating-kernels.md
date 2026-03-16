# Integrating kernels

This page shows how different projects use `kernels`.

## autoresearch

[karpathy/autoresearch](https://github.com/karpathy/autoresearch) [uses](https://github.com/karpathy/autoresearch/blob/c2450add72cc80317be1fe8111974b892da10944/train.py#L23) `kernels` to
integrate Flash-Attention 3 through the `get_kernes()` method.

## AReaL

[inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) uses `kernels` in an opt-in manner to integrate
optimized attention mechanisms.

## transformers

[huggingface/transformers](https://github.com/huggingface/transformers/) primarily
depends on `kernels` for all optimizations related to optimized kernels, including
optimized attention implementations, MOE blocks, and quantization. Besides
`get_kernel()`, it also [layers](./layers.md) to optimize the forward passes
of common layers involved in the modeling blocks. Some references are available
[here](<https://github.com/search?q=repo%3Ahuggingface%2Ftransformers%20get_kernel(&type=code>)
and [here](https://github.com/search?q=repo%3Ahuggingface%2Ftransformers+use_kernel_forward_from_hub&type=code).

Refer to the following posts to know more:

* [Tricks from OpenAI gpt-oss YOU 🫵 can use with transformers](https://huggingface.co/blog/faster-transformers)
* [Mixture of Experts (MoEs) in Transformers](https://huggingface.co/blog/moe-transformers)

## diffusers

Similar to `transformers`, [huggingface/diffusers](https://github.com/huggingface/diffusers/) uses
`kernels` for integrating optimized kernels to [compute attention](https://github.com/huggingface/diffusers/blob/e5aa719241f9b74d6700be3320a777799bfab70a/src/diffusers/models/attention_dispatch.py).