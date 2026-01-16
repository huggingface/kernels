import torch

from kernels.benchmark import Benchmark


def _extract_output(result):
    """Extract tensor from result (handles both tensor and tuple returns)."""
    if isinstance(result, tuple):
        return result[0]
    return result


def _reference_attention(query, key, value, causal=False):
    """Reference implementation using PyTorch SDPA."""
    query, key, value = (x.transpose(1, 2).contiguous() for x in (query, key, value))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=causal
        )
    return out.transpose(1, 2).contiguous()


def _varlen_reference_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=False):
    """Reference implementation for variable length attention."""
    batch_size = cu_seqlens_q.shape[0] - 1
    total_tokens_q = q.shape[0]
    out = torch.zeros(
        (total_tokens_q, q.shape[1], q.shape[2]), device=q.device, dtype=q.dtype
    )

    for b in range(batch_size):
        start_q, end_q = cu_seqlens_q[b], cu_seqlens_q[b + 1]
        start_k, end_k = cu_seqlens_k[b], cu_seqlens_k[b + 1]

        q_slice = q[start_q:end_q].unsqueeze(0)
        k_slice = k[start_k:end_k].unsqueeze(0)
        v_slice = v[start_k:end_k].unsqueeze(0)

        attn_out = _reference_attention(q_slice, k_slice, v_slice, causal=causal)
        out[start_q:end_q] = attn_out.squeeze(0)

    return out


class FlashAttentionBenchmark(Benchmark):
    seed: int = 42

    # Workload: small (B=2, S=128, H=8, D=64)
    def setup_small(self):
        B, S, H, D = 2, 128, 8, 64
        self.q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.out = torch.empty(B, S, H, D, device="cuda", dtype=torch.float16)

    def benchmark_small(self):
        self.out = _extract_output(
            self.kernel.flash_attn_func(self.q, self.k, self.v, causal=False)
        )

    def verify_small(self) -> torch.Tensor:
        return _reference_attention(self.q, self.k, self.v, causal=False)

    # Workload: medium (B=4, S=512, H=16, D=64)
    def setup_medium(self):
        B, S, H, D = 4, 512, 16, 64
        self.q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.out = torch.empty(B, S, H, D, device="cuda", dtype=torch.float16)

    def benchmark_medium(self):
        self.out = _extract_output(
            self.kernel.flash_attn_func(self.q, self.k, self.v, causal=False)
        )

    def verify_medium(self) -> torch.Tensor:
        return _reference_attention(self.q, self.k, self.v, causal=False)

    # Workload: large (B=8, S=1024, H=32, D=128)
    def setup_large(self):
        B, S, H, D = 8, 1024, 32, 128
        self.q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.out = torch.empty(B, S, H, D, device="cuda", dtype=torch.float16)

    def benchmark_large(self):
        self.out = _extract_output(
            self.kernel.flash_attn_func(self.q, self.k, self.v, causal=False)
        )

    def verify_large(self) -> torch.Tensor:
        return _reference_attention(self.q, self.k, self.v, causal=False)


class FlashAttentionCausalBenchmark(Benchmark):
    seed: int = 42

    # Workload: small (B=2, S=128, H=8, D=64)
    def setup_small(self):
        B, S, H, D = 2, 128, 8, 64
        self.q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.out = torch.empty(B, S, H, D, device="cuda", dtype=torch.float16)

    def benchmark_small(self):
        self.out = _extract_output(
            self.kernel.flash_attn_func(self.q, self.k, self.v, causal=True)
        )

    def verify_small(self) -> torch.Tensor:
        return _reference_attention(self.q, self.k, self.v, causal=True)

    # Workload: medium (B=4, S=512, H=16, D=64)
    def setup_medium(self):
        B, S, H, D = 4, 512, 16, 64
        self.q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.out = torch.empty(B, S, H, D, device="cuda", dtype=torch.float16)

    def benchmark_medium(self):
        self.out = _extract_output(
            self.kernel.flash_attn_func(self.q, self.k, self.v, causal=True)
        )

    def verify_medium(self) -> torch.Tensor:
        return _reference_attention(self.q, self.k, self.v, causal=True)

    # Workload: large (B=8, S=1024, H=32, D=128)
    def setup_large(self):
        B, S, H, D = 8, 1024, 32, 128
        self.q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.out = torch.empty(B, S, H, D, device="cuda", dtype=torch.float16)

    def benchmark_large(self):
        self.out = _extract_output(
            self.kernel.flash_attn_func(self.q, self.k, self.v, causal=True)
        )

    def verify_large(self) -> torch.Tensor:
        return _reference_attention(self.q, self.k, self.v, causal=True)


class FlashAttentionVarlenBenchmark(Benchmark):
    seed: int = 42

    # Workload: small (3 sequences, max_seqlen=64)
    def setup_small(self):
        H, D = 8, 64
        # Pack sequences of lengths [32, 48, 64]
        seqlens = [32, 48, 64]
        total = sum(seqlens)
        self.q = torch.randn(total, H, D, device="cuda", dtype=torch.float16)
        self.k = torch.randn(total, H, D, device="cuda", dtype=torch.float16)
        self.v = torch.randn(total, H, D, device="cuda", dtype=torch.float16)
        self.cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seqlens), 0)),
            device="cuda",
            dtype=torch.int32,
        )
        self.max_seqlen = max(seqlens)
        self.out = torch.empty(total, H, D, device="cuda", dtype=torch.float16)

    def benchmark_small(self):
        self.out = _extract_output(
            self.kernel.flash_attn_varlen_func(
                self.q,
                self.k,
                self.v,
                self.cu_seqlens,
                self.cu_seqlens,
                self.max_seqlen,
                self.max_seqlen,
            )
        )

    def verify_small(self) -> torch.Tensor:
        return _varlen_reference_attention(
            self.q, self.k, self.v, self.cu_seqlens, self.cu_seqlens, causal=False
        )

    # Workload: medium (5 sequences, max_seqlen=256)
    def setup_medium(self):
        H, D = 16, 64
        seqlens = [128, 192, 256, 200, 150]
        total = sum(seqlens)
        self.q = torch.randn(total, H, D, device="cuda", dtype=torch.float16)
        self.k = torch.randn(total, H, D, device="cuda", dtype=torch.float16)
        self.v = torch.randn(total, H, D, device="cuda", dtype=torch.float16)
        self.cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seqlens), 0)),
            device="cuda",
            dtype=torch.int32,
        )
        self.max_seqlen = max(seqlens)
        self.out = torch.empty(total, H, D, device="cuda", dtype=torch.float16)

    def benchmark_medium(self):
        self.out = _extract_output(
            self.kernel.flash_attn_varlen_func(
                self.q,
                self.k,
                self.v,
                self.cu_seqlens,
                self.cu_seqlens,
                self.max_seqlen,
                self.max_seqlen,
            )
        )

    def verify_medium(self) -> torch.Tensor:
        return _varlen_reference_attention(
            self.q, self.k, self.v, self.cu_seqlens, self.cu_seqlens, causal=False
        )

    # Workload: large (8 sequences, max_seqlen=512)
    def setup_large(self):
        H, D = 32, 128
        seqlens = [256, 384, 512, 448, 320, 480, 400, 512]
        total = sum(seqlens)
        self.q = torch.randn(total, H, D, device="cuda", dtype=torch.float16)
        self.k = torch.randn(total, H, D, device="cuda", dtype=torch.float16)
        self.v = torch.randn(total, H, D, device="cuda", dtype=torch.float16)
        self.cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seqlens), 0)),
            device="cuda",
            dtype=torch.int32,
        )
        self.max_seqlen = max(seqlens)
        self.out = torch.empty(total, H, D, device="cuda", dtype=torch.float16)

    def benchmark_large(self):
        self.out = _extract_output(
            self.kernel.flash_attn_varlen_func(
                self.q,
                self.k,
                self.v,
                self.cu_seqlens,
                self.cu_seqlens,
                self.max_seqlen,
                self.max_seqlen,
            )
        )

    def verify_large(self) -> torch.Tensor:
        return _varlen_reference_attention(
            self.q, self.k, self.v, self.cu_seqlens, self.cu_seqlens, causal=False
        )
