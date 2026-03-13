# Kernel-Agent 项目知识提取

本文档记录从 `/home/jixiong/kernel-agent` 项目中提取的核心知识，作为 ROCm kernel skills 的基础。

## 1. 项目概况

kernel-agent 是一个 **LLM 驱动的 Triton/Helion kernel 生成与评测工作流**，专门面向 AMD ROCm 平台。

| 组件 | 说明 |
|------|------|
| **后端** | Triton (主要), Helion (实验性) |
| **目标平台** | AMD GPU (ROCm) |
| **评测基准** | KernelBench (Level 1-7) |
| **工作流** | 生成 → 执行 → 正确性检查 → 性能优化 → 迭代 |
| **LLM 提供商** | OpenAI, Anthropic, Google, AMD on-prem |

## 2. AMD GPU 硬件参数 (来自 amd_gpu_specs.py)

### MI355X (gfx950) - CDNA3+

| 参数 | 值 | 优化影响 |
|------|-----|---------|
| GPU 架构 | CDNA3+ (gfx950) | 编译目标 |
| GPU 显存 | 288GB HBM3e | 大模型无压力 |
| 内存带宽 | 8 TB/s | 内存受限 kernel 的上限 |
| XCD 数量 | **32** | XCD Swizzle 必须用 NUM_XCDS=32 |
| CU 总数 | 256 | Grid 大小的倍数 |
| 每 XCD 的 CU | 8 | XCD 间负载均衡 |
| LDS/CU | **160 KB** | 比 MI300X 大 2.5 倍 |
| L2 Cache | 256 MB | 大型共享缓存 |
| Wavefront | 64 | CDNA 固定 Wave64 |
| MFMA 指令 | 16x16 (最优), 32x32 | matrix_instr_nonkdim=16 |
| FP8 格式 | float8_e4m3fn (OCP) | 与 MI300X 不同！ |
| 最优 num_warps | 4-16 | autotune 范围 |
| 最优 num_stages | 2-3 | 避免 LDS 溢出 |
| 最优 BLOCK_SIZE (1D) | 1024-8192 | 比 MI300X 更大 |
| 最优 BLOCK_M/N (2D) | 128-256 | GEMM tile 大小 |

### R9700 (RDNA4, gfx1201)

| 参数 | 值 | 优化影响 |
|------|-----|---------|
| GPU 架构 | RDNA4 (gfx1201) | Wave32 模式 |
| Wavefront | **32** | 归约代码需要不同偏移 |
| CU 总数 | 64 | Grid 大小的倍数 |
| LDS/CU | 64 KB | 标准大小 |
| L1 Cache | 32 KB | 每 CU 私有 |
| L2 Cache | 8 MB | 全 CU 共享 |
| L3 Cache | 64 MB | 末级缓存 |
| Cacheline | **256 B** | 比 RDNA3 更大，需更严格对齐 |
| Max Threads/Block | 1024 | 32 waves × 32 threads |
| Max Threads/CU | 2048 | 64 waves × 32 threads |
| FP16 矩阵 TFLOPS | 191 | 矩阵指令 |
| FP8 矩阵 TFLOPS | 383 | 推理加速 |
| 矩阵核心 | 有限 (无 FP8 MFMA) | 不支持高级矩阵指令 |

## 3. 关键优化知识 (来自 prompt_constructor.py)

### MI355X 必须的优化

1. **XCD Swizzle (GEMM 必须)**: NUM_XCDS=32，将 block ID 映射到 32 个 XCD
2. **L2 Cache Grouping**: GROUP_M=8 或 16，提高 L2 缓存命中率
3. **MFMA 16x16**: matrix_instr_nonkdim=16
4. **环境变量**: `TRITON_HIP_USE_BLOCK_PINGPONG=1`, `TRITON_HIP_USE_ASYNC_COPY=1`
5. **num_stages=2-3**: 避免 LDS 溢出

### Triton on ROCm 禁忌

- **禁止** `tl.libdevice.*` (CUDA 专属)
- **禁止** `tl.tanh` (不支持，用 `(exp(2x)-1)/(exp(2x)+1)`)
- **禁止** `break/continue` (用 `tl.where` 替代)
- **禁止** Python `min()/max()` (用 `tl.minimum()/tl.maximum()`)
- **必须** 用 `tl.float32` 做累加器
- **必须** 对 exp/log/sqrt/rsqrt/除法 转换为 FP32

### Autotune 配置

#### 逐元素 (1D)

**MI355X**: BLOCK_SIZE = [1024, 2048, 4096, 4096, 8192, 16384]
**R9700**: BLOCK_SIZE = [256, 512, 1024] (更小)

#### GEMM (2D)

**MI355X**: BLOCK_M/N = [128-256], BLOCK_K = [32-64], GROUP_M = 8

## 4. 问题分类体系 (来自 classify_problem)

| 类别 | 匹配模式 | 典型算子 |
|------|---------|---------|
| elementwise | relu, gelu, swish, silu, sigmoid, tanh, elu... | 激活函数 |
| softmax | softmax, logsoftmax | Softmax 变体 |
| norm | layernorm, batchnorm, rmsnorm, groupnorm... | 归一化 |
| pooling | pool | 池化操作 |
| reduction | sum_reduction, mean_reduction, max_reduction... | 归约操作 |
| attention | attention, multihead | 注意力机制 |
| matvec | matrix_vector, matvec | 矩阵-向量乘 |
| batched_gemm | batch, bmm | 批量矩阵乘 |
| gemm_2d | matmul, gemm, mm_ | 2D 矩阵乘 |

## 5. KernelBench 测试结果关键发现

### 表现优秀的类别 (在 kernel-agent 上)

| 类别 | 最佳 Speedup | 代表算子 |
|------|-------------|---------|
| Reduction | 5.00x | Sum reduction |
| Pooling | 5.16x | Average Pooling 3D |
| 激活函数 | 2.94x | Softsign, Softplus, Swish |
| 归一化 | 1.73x | LayerNorm |
| 特殊 GEMM | 1.98x | 对角矩阵乘 |

### 需要重点优化的类别

| 类别 | 当前 Speedup | 根本原因 |
|------|-------------|---------|
| 大 K GEMM | 0.04x | 寄存器压力、内存访问不优 |
| BatchNorm | 0.04x | HIP 运行时错误、同步问题 |
| 对称/三角矩阵乘 | 0.08-0.20x | 线程利用率低 |
| Argmax/Argmin | FAILED | Triton API 限制 |
| 融合算子 | 0.32x (平均) | 多操作组合复杂度 |

### 常见错误类型

1. **HIP Runtime Error**: GPU 内存访问冲突
2. **精度问题**: FP16 累积误差
3. **program_id 限制**: 3D Grid 映射
4. **tl.store() kwarg 错误**: Triton API 差异
5. **max_contiguous 错误**: 内存访问模式

## 6. 性能分析工具链

| 工具 | 用途 |
|------|------|
| `rocprof` / `rocprofv3` | GPU kernel profiling |
| `rocm-bandwidth-test` | 内存带宽测试 |
| `rocminfo` | GPU 设备信息 |
| `rocm-smi` | GPU 状态监控 |
| `omniperf` | 全面性能分析 |
| `omnitrace` | 系统级追踪 |
