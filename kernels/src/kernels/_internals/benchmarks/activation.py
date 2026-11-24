import torch
import torch.nn.functional as F

from kernels._internals.benchmark import Benchmark

DECODE_TOKENS = 1
PREFILL_TOKENS = 2048


class _GatedActivationBenchmark(Benchmark):
    seed: int = 42

    # The input is the output of an MLP's gate and up projections.
    def _setup(self, tokens: int, intermediate_size: int, model: str):
        self.x = torch.randn(tokens, 2 * intermediate_size, device=self.device, dtype=torch.float16)
        self.out = torch.empty(tokens, intermediate_size, device=self.device, dtype=torch.float16)
        self.params = {"shape": list(self.x.shape), "dtype": "float16", "model": model}

    # Each activation calls its kernel on `x` into `out`.
    def _run(self):
        raise NotImplementedError

    # Each activation returns the PyTorch result to compare against.
    def _reference(self) -> torch.Tensor:
        raise NotImplementedError

    # Each model has a decode workload (one token) and a prefill workload (a chunk of the prompt).
    # The width is the model's MLP intermediate size.

    # Qwen3-0.6B, intermediate size 3072.
    def setup_qwen3_0_6b_decode(self):
        self._setup(DECODE_TOKENS, 3072, "Qwen3-0.6B")

    def benchmark_qwen3_0_6b_decode(self):
        self._run()

    def verify_qwen3_0_6b_decode(self) -> torch.Tensor:
        return self._reference()

    def setup_qwen3_0_6b_prefill(self):
        self._setup(PREFILL_TOKENS, 3072, "Qwen3-0.6B")

    def benchmark_qwen3_0_6b_prefill(self):
        self._run()

    def verify_qwen3_0_6b_prefill(self) -> torch.Tensor:
        return self._reference()

    # Qwen3-8B, intermediate size 12288.
    def setup_qwen3_8b_decode(self):
        self._setup(DECODE_TOKENS, 12288, "Qwen3-8B")

    def benchmark_qwen3_8b_decode(self):
        self._run()

    def verify_qwen3_8b_decode(self) -> torch.Tensor:
        return self._reference()

    def setup_qwen3_8b_prefill(self):
        self._setup(PREFILL_TOKENS, 12288, "Qwen3-8B")

    def benchmark_qwen3_8b_prefill(self):
        self._run()

    def verify_qwen3_8b_prefill(self) -> torch.Tensor:
        return self._reference()

    # Qwen3-32B, intermediate size 25600.
    def setup_qwen3_32b_decode(self):
        self._setup(DECODE_TOKENS, 25600, "Qwen3-32B")

    def benchmark_qwen3_32b_decode(self):
        self._run()

    def verify_qwen3_32b_decode(self) -> torch.Tensor:
        return self._reference()

    def setup_qwen3_32b_prefill(self):
        self._setup(PREFILL_TOKENS, 25600, "Qwen3-32B")

    def benchmark_qwen3_32b_prefill(self):
        self._run()

    def verify_qwen3_32b_prefill(self) -> torch.Tensor:
        return self._reference()


class SiluAndMulBenchmark(_GatedActivationBenchmark):
    def _run(self):
        self.kernel.silu_and_mul(self.out, self.x)

    def _reference(self) -> torch.Tensor:
        d = self.x.shape[-1] // 2
        return F.silu(self.x[..., :d]) * self.x[..., d:]


class GeluAndMulBenchmark(_GatedActivationBenchmark):
    def _run(self):
        self.kernel.gelu_and_mul(self.out, self.x)

    def _reference(self) -> torch.Tensor:
        d = self.x.shape[-1] // 2
        return F.gelu(self.x[..., :d]) * self.x[..., d:]
