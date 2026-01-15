import torch
import torch.nn.functional as F

from kernels.benchmark import Benchmark


class SiluAndMulBenchmark(Benchmark):
    seed: int = 42

    # Workload: small
    def setup_small(self):
        self.x = torch.randn(1, 128, 512, device="cuda", dtype=torch.float16)
        self.out = torch.empty(1, 128, 256, device="cuda", dtype=torch.float16)

    def benchmark_small(self):
        self.kernel.silu_and_mul(self.out, self.x)

    def verify_small(self) -> torch.Tensor:
        d = self.x.shape[-1] // 2
        return F.silu(self.x[..., :d]) * self.x[..., d:]

    # Workload: medium
    def setup_medium(self):
        self.x = torch.randn(4, 512, 1024, device="cuda", dtype=torch.float16)
        self.out = torch.empty(4, 512, 512, device="cuda", dtype=torch.float16)

    def benchmark_medium(self):
        self.kernel.silu_and_mul(self.out, self.x)

    def verify_medium(self) -> torch.Tensor:
        d = self.x.shape[-1] // 2
        return F.silu(self.x[..., :d]) * self.x[..., d:]

    # Workload: large
    def setup_large(self):
        self.x = torch.randn(8, 1024, 2048, device="cuda", dtype=torch.float16)
        self.out = torch.empty(8, 1024, 1024, device="cuda", dtype=torch.float16)

    def benchmark_large(self):
        self.kernel.silu_and_mul(self.out, self.x)

    def verify_large(self) -> torch.Tensor:
        d = self.x.shape[-1] // 2
        return F.silu(self.x[..., :d]) * self.x[..., d:]
