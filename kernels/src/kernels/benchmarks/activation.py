import torch
import torch.nn.functional as F

from kernels.benchmark import Benchmark


class SiluAndMulBenchmark(Benchmark):
    seed: int = 42

    # Workload: small
    def setup_small(self):
        self.x = torch.randn(8, 1024, 2048, device=self.device, dtype=torch.float16)
        self.out = torch.empty(8, 1024, 1024, device=self.device, dtype=torch.float16)

    def benchmark_small(self):
        self.kernel.silu_and_mul(self.out, self.x)

    def verify_small(self) -> torch.Tensor:
        d = self.x.shape[-1] // 2
        return F.silu(self.x[..., :d]) * self.x[..., d:]

    # Workload: medium
    def setup_medium(self):
        self.x = torch.randn(8, 2048, 4096, device=self.device, dtype=torch.float16)
        self.out = torch.empty(8, 2048, 2048, device=self.device, dtype=torch.float16)

    def benchmark_medium(self):
        self.kernel.silu_and_mul(self.out, self.x)

    def verify_medium(self) -> torch.Tensor:
        d = self.x.shape[-1] // 2
        return F.silu(self.x[..., :d]) * self.x[..., d:]

    # Workload: large
    def setup_large(self):
        self.x = torch.randn(8, 4096, 8192, device=self.device, dtype=torch.float16)
        self.out = torch.empty(8, 4096, 4096, device=self.device, dtype=torch.float16)

    def benchmark_large(self):
        self.kernel.silu_and_mul(self.out, self.x)
        self.kernel.silu_and_mul(self.out, self.x)

    def verify_large(self) -> torch.Tensor:
        d = self.x.shape[-1] // 2
        return F.silu(self.x[..., :d]) * self.x[..., d:]


class GeluAndMulBenchmark(Benchmark):
    seed: int = 42

    # Workload: small
    def setup_small(self):
        self.x = torch.randn(8, 1024, 2048, device=self.device, dtype=torch.float16)
        self.out = torch.empty(8, 1024, 1024, device=self.device, dtype=torch.float16)

    def benchmark_small(self):
        self.kernel.gelu_and_mul(self.out, self.x)

    def verify_small(self) -> torch.Tensor:
        d = self.x.shape[-1] // 2
        return F.gelu(self.x[..., :d]) * self.x[..., d:]

    # Workload: medium
    def setup_medium(self):
        self.x = torch.randn(8, 2048, 4096, device=self.device, dtype=torch.float16)
        self.out = torch.empty(8, 2048, 2048, device=self.device, dtype=torch.float16)

    def benchmark_medium(self):
        self.kernel.gelu_and_mul(self.out, self.x)

    def verify_medium(self) -> torch.Tensor:
        d = self.x.shape[-1] // 2
        return F.gelu(self.x[..., :d]) * self.x[..., d:]

    # Workload: large
    def setup_large(self):
        self.x = torch.randn(8, 4096, 8192, device=self.device, dtype=torch.float16)
        self.out = torch.empty(8, 4096, 4096, device=self.device, dtype=torch.float16)

    def benchmark_large(self):
        self.kernel.gelu_and_mul(self.out, self.x)

    def verify_large(self) -> torch.Tensor:
        d = self.x.shape[-1] // 2
        return F.gelu(self.x[..., :d]) * self.x[..., d:]
