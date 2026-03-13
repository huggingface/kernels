# Skill 评估与优化方法论

本文档描述如何系统性地评估和优化 ROCm kernel skills 的质量。

## 1. Skill 质量评估框架

### 1.1 评估维度

| 维度 | 权重 | 衡量标准 | 评估方法 |
|------|------|---------|---------|
| **正确性** | 40% | AI 生成的 kernel 能通过正确性测试 | KernelBench accuracy ratio |
| **性能** | 30% | 生成 kernel 的 speedup 相比 PyTorch baseline | KernelBench speedup ratio |
| **可运行率** | 20% | kernel 能成功编译并运行 | KernelBench runnable ratio |
| **触发准确率** | 10% | AI 在正确场景下使用了正确的 skill | 人工评审 |

### 1.2 KernelBench 自动评估

```bash
# 运行 Level 1 全量评估
python -m kernel_agent.evaluation.kernelbench_success_evaluator \
    --config examples/workflows/evaluation/kernelbench/config_eval.yml \
    --dataset datasets/kernelbench/kernel_bench_level_1.json

# 输出指标
# - runnable_ratio: 可运行比率
# - accuracy_ratio: 正确性比率  
# - speed_ratio: 达到 speedup > 1.0x 的比率
```

### 1.3 逐类别评估

对每个 operator 类别单独评估：

```bash
# Level 1 Selected (按类别)
python -m kernel_agent.evaluation.kernelbench_success_evaluator \
    --config examples/workflows/evaluation/kernelbench/config_eval_level1_selected.yml
```

**评估记录表模板：**

| 类别 | 算子数 | 可运行 | 正确 | Speedup>1x | 平均 Speedup | Skill 版本 |
|------|--------|--------|------|-----------|-------------|-----------|
| GEMM | 18 | ?/18 | ?/18 | ?/18 | ?x | v0.1 |
| Elementwise | 14 | ?/14 | ?/14 | ?/14 | ?x | v0.1 |
| Normalization | 8 | ?/8 | ?/8 | ?/8 | ?x | v0.1 |
| ... | ... | ... | ... | ... | ... | ... |

## 2. 迭代优化流程

### 2.1 PDCA 循环

```
Plan  → 识别性能瓶颈或失败模式
Do    → 修改 SKILL.md 或 references 中的指引
Check → 重新运行 KernelBench 评估
Act   → 确认改进，合并到 skill；或回滚
```

### 2.2 具体优化步骤

#### Step 1: 收集失败案例

```python
# 从评估结果中提取失败案例
# 分析每个失败的原因类型：
# - compilation_error: 编译错误 → 修改模板代码
# - runtime_error: 运行时错误 → 添加 troubleshooting 条目
# - accuracy_error: 精度问题 → 修改精度相关指引
# - performance_low: 性能不达标 → 添加优化策略
```

#### Step 2: 根因分析

| 失败类型 | 检查项 | 修改位置 |
|---------|--------|---------|
| tl.libdevice 错误 | 是否遗漏了 ROCm 禁忌 | SKILL.md "Critical ROCm Constraints" |
| LDS 溢出 | num_stages 建议是否正确 | GPU optimization guide |
| GEMM 极慢 | 是否缺少 XCD swizzle | kernel-templates.md Template 5 (GEMM with XCD Swizzle) |
| 精度不达标 | FP32 累加是否到位 | kernel-templates.md 所有模板 |
| Python min/max | 是否提醒了 tl.minimum | troubleshooting.md |

#### Step 3: 修改 Skill 内容

按照失败原因修改对应文件：
- **模式错误** → 修改 `kernel-templates.md`
- **知识缺失** → 修改 `SKILL.md` 或 GPU 优化指南
- **新陷阱** → 添加到 `troubleshooting.md`
- **性能数据过时** → 更新 benchmark 表格

#### Step 4: A/B 测试

```bash
# Version A: 原始 skill
cp -r rocm-kernels rocm-kernels-v1

# Version B: 修改后的 skill
# (直接编辑 rocm-kernels/)

# 分别运行评估，对比结果
# Compare: runnable_ratio, accuracy_ratio, speed_ratio
```

## 3. 与 AI 协作优化 Skill 的方法

### 3.1 "生成-评测-反馈" 循环

```
你 (编写 Skill)
  ↓
AI (使用 Skill 生成 kernel)
  ↓
KernelBench (评测 kernel)
  ↓
你 (分析失败，改进 Skill)
  ↓
(重复)
```

### 3.2 具体协作方式

#### 方式 1: 让 AI 帮你分析失败

```
提示词: "这是 KernelBench Level 1 的评测结果 [粘贴结果]。
请分析以下失败案例的根因，并建议修改 SKILL.md 的哪些部分。"
```

#### 方式 2: 让 AI 帮你生成测试用例

```
提示词: "根据 rocm-kernels skill，为 GEMM 类别生成一个测试 kernel，
目标是 4096x4096 方阵乘法，使用 MI355X 优化。"
```

然后手动运行测试，看结果是否符合预期。

#### 方式 3: 让 AI 帮你补充 Skill 内容

```
提示词: "BatchNorm 在 AMD GPU 上的评测结果是 0.04x（极差）。
错误类型是 HIP 运行时错误。请帮我分析原因，并在 troubleshooting.md
中添加对应的解决方案。"
```

### 3.3 Skill 版本管理

```
rocm-kernels/
├── SKILL.md              # 主文件 (跟踪版本号)
├── CHANGELOG.md          # 变更日志 (每次优化后记录)
├── references/
└── scripts/
```

**CHANGELOG 格式：**

```markdown
## v0.2 (2026-03-15)
- Added XCD swizzle pattern for GEMM (fixed 0.3x → 1.1x speedup)
- Added tanh workaround for ROCm
- Fixed LDS overflow guidance for MI355X

## v0.1 (2026-03-10)
- Initial version with basic templates
```

## 4. KernelBench 分类 Skill 开发计划

### 4.1 开发优先级

| 优先级 | 类别 | 原因 | 预期 Skill 文件 |
|--------|------|------|----------------|
| **P0** | Elementwise | 最多算子、最容易成功 | `elementwise-skill.md` |
| **P0** | GEMM | 最高影响、最频繁使用 | `gemm-skill.md` |
| **P1** | Normalization | 常用、中等难度 | `normalization-skill.md` |
| **P1** | Reduction | 常用、有成熟模式 | `reduction-skill.md` |
| **P1** | Softmax | Attention 基础 | `softmax-skill.md` |
| **P2** | Pooling | 中等频率、Grid 映射挑战 | `pooling-skill.md` |
| **P2** | Attention | 高复杂度、高价值 | `attention-skill.md` |
| **P2** | Fused | 多操作组合 | `fused-skill.md` |

### 4.2 每个 Skill 文件结构

```markdown
---
name: rocm-{category}-kernel
description: "..."
---

# {Category} Kernel Skill

## Pattern Overview
[核心算法模式]

## Template Code
[可复制的完整代码]

## Autotune Configurations
[MI355X 和 R9700 的推荐配置]

## Common Mistakes
[该类别特有的陷阱]

## Benchmark Results
[该类别的已知性能数据]
```

### 4.3 评估指标目标

| 类别 | 可运行率目标 | 正确率目标 | 平均 Speedup 目标 |
|------|------------|-----------|------------------|
| Elementwise | >95% | >90% | >1.5x |
| GEMM | >80% | >70% | >0.8x |
| Normalization | >85% | >80% | >1.0x |
| Reduction | >90% | >85% | >1.5x |
| Softmax | >85% | >80% | >1.0x |
| Pooling | >70% | >60% | >1.0x |
| Attention | >60% | >50% | >0.8x |
| Fused | >50% | >40% | >0.8x |

## 5. 持续监控

### 5.1 定期评估

- **每周**: 运行一次 Level 1 全量评估
- **每次 Skill 修改后**: 运行修改类别的评估
- **每月**: 运行 Level 1+2 全量评估

### 5.2 回归检测

修改 Skill 后，确保不会导致其他类别的性能下降：

```bash
# 修改前: 保存基线
python eval.py --save-baseline baseline_v1.json

# 修改后: 对比
python eval.py --compare-baseline baseline_v1.json
# 任何类别的指标下降 > 5% 需要调查
```

## 6. 总结：优化 Skill 的核心原则

1. **数据驱动**: 每次修改都基于 KernelBench 评估数据
2. **分类优化**: 按 operator 类别独立迭代
3. **最小修改**: 每次只改一个点，方便归因
4. **版本记录**: 每次修改记录 CHANGELOG
5. **A/B 测试**: 对比修改前后的评测结果
6. **渐进式**: 先覆盖高优先级类别（Elementwise → GEMM → Norm）
7. **陷阱文档化**: 每个新发现的坑都写入 troubleshooting.md
