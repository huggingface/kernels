import torch

from kernels.benchmark import Benchmark


class RMSNormBenchmark(Benchmark):
    seed: int = 42
    eps: float = 1e-5

    # Workload: small (B=2, S=128, D=768)
    def setup_small(self):
        B, S, D = 2, 128, 768
        self.x = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
        self.weight = torch.ones(D, device="cuda", dtype=torch.float16)
        self.out = torch.empty_like(self.x)
        self.B, self.S, self.D = B, S, D

    def benchmark_small(self):
        self.out = self.kernel.dropout_add_ln_fwd(
            input=self.x.view(-1, self.D),
            gamma=self.weight,
            beta=None,
            rowscale=None,
            colscale=None,
            x0_subset=None,
            z_subset=None,
            dropout_p=0.0,
            epsilon=self.eps,
            rowscale_const=1.0,
            z_numrows=self.S,
            gen=None,
            residual_in_fp32=False,
            is_rms_norm=True,
        )[0].view(self.B, self.S, self.D)

    def verify_small(self) -> torch.Tensor:
        var = self.x.pow(2).mean(-1, keepdim=True)
        return (self.x * torch.rsqrt(var + self.eps)) * self.weight

    # Workload: medium (B=4, S=512, D=2048)
    def setup_medium(self):
        B, S, D = 4, 512, 2048
        self.x = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
        self.weight = torch.ones(D, device="cuda", dtype=torch.float16)
        self.out = torch.empty_like(self.x)
        self.B, self.S, self.D = B, S, D

    def benchmark_medium(self):
        self.out = self.kernel.dropout_add_ln_fwd(
            input=self.x.view(-1, self.D),
            gamma=self.weight,
            beta=None,
            rowscale=None,
            colscale=None,
            x0_subset=None,
            z_subset=None,
            dropout_p=0.0,
            epsilon=self.eps,
            rowscale_const=1.0,
            z_numrows=self.S,
            gen=None,
            residual_in_fp32=False,
            is_rms_norm=True,
        )[0].view(self.B, self.S, self.D)

    def verify_medium(self) -> torch.Tensor:
        var = self.x.pow(2).mean(-1, keepdim=True)
        return (self.x * torch.rsqrt(var + self.eps)) * self.weight

    # Workload: large (B=8, S=1024, D=4096)
    def setup_large(self):
        B, S, D = 8, 1024, 4096
        self.x = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
        self.weight = torch.ones(D, device="cuda", dtype=torch.float16)
        self.out = torch.empty_like(self.x)
        self.B, self.S, self.D = B, S, D

    def benchmark_large(self):
        self.out = self.kernel.dropout_add_ln_fwd(
            input=self.x.view(-1, self.D),
            gamma=self.weight,
            beta=None,
            rowscale=None,
            colscale=None,
            x0_subset=None,
            z_subset=None,
            dropout_p=0.0,
            epsilon=self.eps,
            rowscale_const=1.0,
            z_numrows=self.S,
            gen=None,
            residual_in_fp32=False,
            is_rms_norm=True,
        )[0].view(self.B, self.S, self.D)

    def verify_large(self) -> torch.Tensor:
        var = self.x.pow(2).mean(-1, keepdim=True)
        return (self.x * torch.rsqrt(var + self.eps)) * self.weight


class LayerNormBenchmark(Benchmark):
    seed: int = 42
    eps: float = 1e-5

    # Workload: small (B=2, S=128, D=768)
    def setup_small(self):
        B, S, D = 2, 128, 768
        self.x = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
        self.weight = torch.ones(D, device="cuda", dtype=torch.float16)
        self.out = torch.empty_like(self.x)
        self.B, self.S, self.D = B, S, D

    def benchmark_small(self):
        self.out = self.kernel.dropout_add_ln_fwd(
            input=self.x.view(-1, self.D),
            gamma=self.weight,
            beta=None,
            rowscale=None,
            colscale=None,
            x0_subset=None,
            z_subset=None,
            dropout_p=0.0,
            epsilon=self.eps,
            rowscale_const=1.0,
            z_numrows=self.S,
            gen=None,
            residual_in_fp32=False,
            is_rms_norm=False,
        )[0].view(self.B, self.S, self.D)

    def verify_small(self) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            self.x, [self.D], self.weight, eps=self.eps
        )

    # Workload: medium (B=4, S=512, D=2048)
    def setup_medium(self):
        B, S, D = 4, 512, 2048
        self.x = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
        self.weight = torch.ones(D, device="cuda", dtype=torch.float16)
        self.out = torch.empty_like(self.x)
        self.B, self.S, self.D = B, S, D

    def benchmark_medium(self):
        self.out = self.kernel.dropout_add_ln_fwd(
            input=self.x.view(-1, self.D),
            gamma=self.weight,
            beta=None,
            rowscale=None,
            colscale=None,
            x0_subset=None,
            z_subset=None,
            dropout_p=0.0,
            epsilon=self.eps,
            rowscale_const=1.0,
            z_numrows=self.S,
            gen=None,
            residual_in_fp32=False,
            is_rms_norm=False,
        )[0].view(self.B, self.S, self.D)

    def verify_medium(self) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            self.x, [self.D], self.weight, eps=self.eps
        )

    # Workload: large (B=8, S=1024, D=4096)
    def setup_large(self):
        B, S, D = 8, 1024, 4096
        self.x = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
        self.weight = torch.ones(D, device="cuda", dtype=torch.float16)
        self.out = torch.empty_like(self.x)
        self.B, self.S, self.D = B, S, D

    def benchmark_large(self):
        self.out = self.kernel.dropout_add_ln_fwd(
            input=self.x.view(-1, self.D),
            gamma=self.weight,
            beta=None,
            rowscale=None,
            colscale=None,
            x0_subset=None,
            z_subset=None,
            dropout_p=0.0,
            epsilon=self.eps,
            rowscale_const=1.0,
            z_numrows=self.S,
            gen=None,
            residual_in_fp32=False,
            is_rms_norm=False,
        )[0].view(self.B, self.S, self.D)

    def verify_large(self) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            self.x, [self.D], self.weight, eps=self.eps
        )
