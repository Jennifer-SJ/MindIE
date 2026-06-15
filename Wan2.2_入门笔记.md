# Wan2.2 视频生成模型入门笔记

---

## 一、项目概览

Wan2.2 是基于 Diffusion Transformer 的视频生成模型，由阿里巴巴 Wan 团队开发，MindIE 在其基础上进行了昇腾 NPU 适配与性能优化。项目支持以下两种核心任务：

- **Text-to-Video (T2V)**：输入文本描述，生成对应视频
- **Image-to-Video (I2V)**：输入图片 + 文本描述，生成视频

项目根目录：`/Users/yujie/MindIE/Wan2.2/`

---

## 二、核心文件结构

```
Wan2.2/
├── generate.py                          # 主入口脚本：参数解析、分布式初始化、模型加载、视频生成
├── wan/
│   ├── __init__.py                      # 模块入口，导出 WanT2V / WanI2V 类
│   ├── text2video.py                    # T2V 管线实现（WanT2V 类）
│   ├── image2video.py                   # I2V 管线实现（WanI2V 类）
│   ├── configs/
│   │   ├── __init__.py                  # 配置注册：WAN_CONFIGS / SIZE_CONFIGS / SUPPORTED_SIZES / OPTIMAL_PARALLEL
│   │   ├── shared_config.py             # 共享配置：T5 dtype、text_len、num_train_timesteps 等
│   │   ├── wan_t2v_A14B.py             # T2V A14B 模型配置：dim=5120, num_heads=40, num_layers=40
│   │   ├── wan_i2v_A14B.py             # I2V A14B 模型配置
│   │   └── wan_ti2v_5B.py              # TI2V 5B 模型配置
│   ├── modules/
│   │   ├── model.py                     # 核心 DiT 模型：WanModel / WanAttentionBlock / WanSelfAttention / WanCrossAttention / Head
│   │   ├── attn_layer.py               # 序列并行注意力层：xFuserLongContextAttention（Ulysses + Ring Attention）
│   │   ├── t5.py                        # T5 文本编码器：T5EncoderModel
│   │   └── vae2_1.py                   # VAE 解码器：Wan2_1_VAE（CausalConv3d）
│   ├── distributed/
│   │   ├── sequence_parallel.py         # 序列并行实现：sp_dit_forward / sp_attn_forward（monkey-patch）
│   │   ├── parallel_mgr.py             # 并行组管理：ParallelConfig / init_parallel_env / CFG/SP/TP Group
│   │   ├── fsdp.py                      # FSDP 分片：shard_model
│   │   ├── tp_applicator.py            # 张量并行应用器：TensorParallelApplicator
│   │   ├── util.py                      # 分布式工具：init_distributed_group
│   │   ├── comm.py                      # 通信原语：all_to_all_4D
│   │   └── group_coordinator.py        # 通信组协调器
│   ├── vae_patch_parallel.py            # VAE 并行：VAE_patch_parallel / set_vae_patch_parallel
│   └── utils/
│       ├── magcache.py                  # MagCache 加速：缓存注意力结果跳过冗余计算
│       ├── rainfusion.py               # Rainfusion v1 稀疏注意力
│       ├── rainfusion_blockwise.py     # Rainfusion v2/v3 块稀疏注意力
│       ├── prompt_extend.py            # Prompt 扩展（DashScope / Qwen）
│       ├── fm_solvers.py               # DPM++ 采样器
│       ├── fm_solvers_unipc.py         # UniPC 采样器
│       └── utils.py                    # 通用工具：save_video / str2bool
```

---

## 三、推理整体流程

### 3.1 完整推理流程图

```
用户输入 Prompt（+ Image）
        │
        ▼
┌─────────────────────────┐
│  1. 参数解析与验证        │  generate.py: _parse_args() + _validate_args()
│     - task, size, frame  │
│     - 分布式参数校验       │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  2. 分布式环境初始化       │  generate.py: generate()
│     - HCCL 后端           │  dist.init_process_group(backend="hccl")
│     - 并行组配置           │  init_parallel_env(parallel_config)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  3. 模型加载              │  wan.WanT2V(config, checkpoint_dir, ...)
│     - T5EncoderModel     │    ├── T5 编码器（文本特征提取）
│     - Wan2_1_VAE         │    ├── VAE 解码器（潜空间→像素）
│     - WanModel × 2       │    ├── low_noise_model（低噪声阶段 DiT）
│       (low + high)       │    └── high_noise_model（高噪声阶段 DiT）
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  4. T5 文本编码           │  text_encoder([input_prompt], device)
│     - 输入：文本字符串     │    → context（条件文本特征）
│     - 输出：文本 embedding │    → context_null（无条件文本特征）
│       shape: [L, C]      │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  5. 初始化噪声            │  torch.randn(target_shape)
│     - shape: [C, F, H, W]│  C=16, F=(frame_num-1)/4+1
│     - 数据类型: float32   │  H/W 由 vae_stride 决定
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  6. 扩散去噪循环（核心）                                      │
│     for t_idx, t in enumerate(timesteps):                    │
│     │                                                        │
│     ├── 6a. 选择模型                                         │
│     │   if t >= boundary: high_noise_model                   │
│     │   else:           low_noise_model                      │
│     │                                                        │
│     ├── 6b. 条件/无条件预测                                   │
│     │   if CFG 并行 (cfg_size=2):                            │
│     │     noise_pred = model(x, t, **arg_all)               │
│     │     → all_gather 分离条件/无条件结果                     │
│     │   else:                                                │
│     │     noise_pred_cond   = model(x, t, **arg_c)          │
│     │     noise_pred_uncond = model(x, t, **arg_null)        │
│     │                                                        │
│     ├── 6c. Classifier-Free Guidance                         │
│     │   noise_pred = uncond + guide_scale × (cond - uncond)  │
│     │                                                        │
│     └── 6d. 采样器步进                                       │
│         latents = scheduler.step(noise_pred, t, latents)     │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│  7. VAE 解码              │  vae.decode(x0)
│     - 潜空间 → 像素空间    │  [C, F, H, W] → [3, N, H×8, W×8]
│     - 支持 VAE 并行        │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  8. 保存视频              │  save_video(videos, save_file)
│     - 输出: MP4 文件       │
└─────────────────────────┘
```

### 3.2 DiT 前向传播内部流程（WanModel.forward / sp_dit_forward）

这是步骤 6 中 `model(x, t, ...)` 的内部细节：

```
输入: x=[C_in, F, H, W], t=时间步, context=文本embedding
        │
        ▼
┌──────────────────────────────┐
│  1. Patch Embedding           │  Conv3d(in_dim=16, dim=5120, kernel=patch_size, stride=patch_size)
│     3D 卷积将视频切分为 patch   │  输入: [1, 16, F, H, W]
│     输出: [B, L, dim]         │  输出: [1, L, 5120]，L = F×H/2×W/2
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  2. 时间步 Embedding          │  sinusoidal_embedding_1d(freq_dim=256, t)
│     - 正弦位置编码             │    → time_embedding: Linear(256→5120) + SiLU + Linear(5120→5120)
│     - 生成调制参数 e0          │    → time_projection: SiLU + Linear(5120→5120×6)
│       shape: [B, L, 6, dim]  │  e0 用于每个 Block 的 AdaLayerNorm 调制
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  3. 文本 Embedding            │  text_embedding: Linear(4096→5120) + GELU + Linear(5120→5120)
│     context: [L_text, 4096]  │  输出: [B, text_len=512, 5120]
│       → [B, 512, 5120]       │
└──────────┬───────────────────┘
           │
           ▼  （序列并行时：x, e, e0 按 rank 切分）
┌──────────────────────────────┐
│  4. 40 层 WanAttentionBlock   │  for b_idx, block in enumerate(self.blocks):
│     ┌────────────────────┐   │
│     │  Block 内部流程:    │   │
│     │                    │   │
│     │  ① AdaLayerNorm    │   │  norm1(x) * (1+e[1]) + e[0]
│     │  ② Self-Attention  │   │  视频token自注意力 + RoPE位置编码
│     │     (WanSelfAttn)  │   │  Q/K/V Linear → RMSNorm → RoPE → FA → O Linear
│     │  ③ 残差连接         │   │  x = x + y * e[2]
│     │  ④ Cross-Attention │   │  视频token × 文本token 交叉注意力
│     │     (WanCrossAttn) │   │  Q from video, K/V from context
│     │  ⑤ FFN             │   │  Linear(5120→13824) + GELU(tanh) + Linear(13824→5120)
│     │  ⑥ 残差连接         │   │  x = x + y * e[5]
│     └────────────────────┘   │
│     × 40 层                   │
└──────────┬───────────────────┘
           │
           ▼  （序列并行时：all_gather 合并 x）
┌──────────────────────────────┐
│  5. Head 输出层               │  norm(x) * (1+e[1]) + e[0] → Linear(5120→16×1×2×2)
│     预测噪声残差               │  输出: [B, L, out_dim×patch_size_product]
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  6. Unpatchify               │  将 patch 序列还原为视频形状
│     [B, L, C] → [C, F, H, W]│
└──────────────────────────────┘
```

### 3.3 双模型机制（low_noise_model / high_noise_model）

Wan2.2 A14B 使用**两个 DiT 模型**分别处理不同噪声阶段：

| 模型 | 处理阶段 | 默认 guide_scale | 判断条件 |
|------|---------|-----------------|---------|
| `high_noise_model` | 高噪声时间步（t ≥ boundary） | 4.0 | `t.item() >= boundary` |
| `low_noise_model` | 低噪声时间步（t < boundary） | 3.0 | `t.item() < boundary` |

- `boundary = 0.875 × num_train_timesteps = 0.875 × 1000 = 875`
- 两个模型结构完全相同（都是 40 层 WanAttentionBlock），但权重不同
- 开启 `offload_model` 时，非活跃模型会被 offload 到 CPU 以节省显存

---

## 四、模型架构详解

### 4.1 WanModel（DiT 主干网络）

定义在 `wan/modules/model.py` 中，是视频生成的核心扩散模型。

**A14B 配置参数（来自 `wan/configs/wan_t2v_A14B.py`）：**

| 参数 | 值 | 含义 |
|------|-----|------|
| `dim` | 5120 | Transformer 隐藏层维度 |
| `ffn_dim` | 13824 | FFN 中间层维度 |
| `num_heads` | 40 | 注意力头数 |
| `num_layers` | 40 | Transformer Block 层数 |
| `freq_dim` | 256 | 时间步正弦编码维度 |
| `text_dim` | 4096 | T5 文本 embedding 输入维度 |
| `text_len` | 512 | 文本 token 固定长度 |
| `in_dim` | 16 | 输入通道数（VAE 潜空间维度） |
| `out_dim` | 16 | 输出通道数 |
| `patch_size` | (1, 2, 2) | 3D patch 大小（时间×高度×宽度） |
| `window_size` | (-1, -1) | 局部注意力窗口（-1 表示全局注意力） |
| `qk_norm` | True | 启用 Q/K 归一化（RMSNorm） |
| `cross_attn_norm` | True | 启用交叉注意力归一化 |
| `param_dtype` | bfloat16 | 模型参数数据类型 |
| `num_train_timesteps` | 1000 | 训练时间步总数 |
| `boundary` | 0.875 | 高/低噪声模型切换阈值 |
| `sample_shift` | 12.0 | 采样偏移因子 |
| `sample_steps` | 40 | 默认推理步数 |
| `sample_guide_scale` | (3.0, 4.0) | 低噪声/高噪声阶段的 CFG 引导强度 |

**WanModel 关键组件：**

```python
class WanModel(ModelMixin, ConfigMixin):
    def __init__(self, ...):
        # 1. Patch Embedding：3D 卷积将视频切分为 token
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # 2. 文本 Embedding：将 T5 输出映射到模型维度
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        # 3. 时间步 Embedding：正弦编码 + MLP
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # 4. 40 层 AttentionBlock
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # 5. 输出头
        self.head = Head(dim, out_dim, patch_size, eps)

        # 6. RoPE 频率
        self.freqs = torch.cat([
            rope_params(1024, d - 4*(d//6)),
            rope_params(1024, 2*(d//6)),
            rope_params(1024, 2*(d//6))
        ], dim=1)
```

### 4.2 WanAttentionBlock（注意力块）

每个 Block 包含以下子模块，执行顺序为：

```
输入 x, e(调制参数), context(文本特征)
        │
        ▼
  ① AdaLayerNorm 调制
     norm1(x) * (1 + e[1]) + e[0]        ← scale & shift
        │
        ▼
  ② Self-Attention（自注意力）
     Q = norm_q(Linear_q(x))              ← Q 经 RMSNorm
     K = norm_k(Linear_k(x))              ← K 经 RMSNorm
     V = Linear_v(x)
     Q, K = RoPE(Q, K)                    ← 旋转位置编码
     Attn = FA(Q, K, V)                   ← Flash Attention
     Out = Linear_o(Attn)
        │
        ▼
  ③ 残差连接
     x = x + Out * e[2]                   ← gate 调制
        │
        ▼
  ④ Cross-Attention（交叉注意力）
     Q = norm_q(Linear_q(x))              ← Q 来自视频
     K = norm_k(Linear_k(context))        ← K 来自文本
     V = Linear_v(context)                ← V 来自文本
     Attn = FA(Q, K, V)
     Out = Linear_o(Attn)
     x = x + Out
        │
        ▼
  ⑤ FFN（前馈网络）
     h = norm2(x) * (1 + e[4]) + e[3]    ← AdaLayerNorm 调制
     y = Linear1(h) → GELU(tanh) → Linear2(y)
     x = x + y * e[5]                     ← gate 调制
        │
        ▼
  输出 x
```

**关键细节：**
- 调制参数 `e0` shape 为 `[B, L, 6, dim]`，被 chunk 为 6 份（e[0]~e[5]），分别用于：
  - e[0], e[1]：Self-Attention 的 scale/shift
  - e[2]：Self-Attention 的 gate
  - e[3], e[4]：FFN 的 scale/shift
  - e[5]：FFN 的 gate
- `modulation` 是可学习参数，shape 为 `[1, 6, dim]`

### 4.3 WanSelfAttention（自注意力）

```python
class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qk_norm, eps):
        self.q = nn.Linear(dim, dim)      # Query 投影
        self.k = nn.Linear(dim, dim)      # Key 投影
        self.v = nn.Linear(dim, dim)      # Value 投影
        self.o = nn.Linear(dim, dim)      # Output 投影
        self.norm_q = WanRMSNorm(dim)     # Q 归一化
        self.norm_k = WanRMSNorm(dim)     # K 归一化
```

**注意力计算路径（根据配置选择不同算子）：**

```
Q, K, V 输入
    │
    ├── Rainfusion v3（稀疏注意力，950 芯片）
    │   └── bsa_sparse_attention_v3（块稀疏注意力 + mask 缓存）
    │
    ├── Rainfusion v2（块稀疏注意力）
    │   └── Rainfusion_blockwise
    │
    ├── Rainfusion v1（空间稀疏注意力，已弃用）
    │   └── Rainfusion
    │
    ├── ALGO=1（高性能 FA 算子，A2/A3 设备）
    │   └── attention_forward(opt_type="ascend_laser_attention")
    │
    ├── ALGO=3（量化 FA）
    │   └── npu_fused_infer_attention_score / fa_quant
    │
    └── 默认（ALGO=0，标准 FA 算子，A5 设备）
        └── attention_forward(opt_type="fused_attn_score")
```

### 4.4 T5EncoderModel（文本编码器）

定义在 `wan/modules/t5.py` 中，使用 UMT5-XXL 模型：

- **模型权重**：`models_t5_umt5-xxl-enc-bf16.pth`
- **Tokenizer**：`google/umt5-xxl`
- **数据类型**：bfloat16
- **输出**：文本 embedding，shape 为 `[text_len=512, text_dim=4096]`
- **支持 FSDP**：通过 `t5_fsdp` 参数开启分片
- **支持 CPU 放置**：通过 `t5_cpu` 参数将 T5 放在 CPU 上运行

### 4.5 Wan2_1_VAE（变分自编码器）

定义在 `wan/modules/vae2_1.py` 中：

- **权重文件**：`Wan2.1_VAE.pth`
- **VAE stride**：`(4, 8, 8)` — 时间维度压缩 4 倍，空间维度各压缩 8 倍
- **潜空间维度**：`z_dim = 16`
- **核心组件**：CausalConv3d（因果 3D 卷积，保证时间维度的因果性）
- **支持并行**：通过 `vae_parallel` 参数开启 VAE patch 并行

**VAE 编码/解码维度变化：**

```
编码：视频 [3, N, H, W] → 潜空间 [16, (N-1)/4+1, H/8, W/8]
解码：潜空间 [16, F, H/8, W/8] → 视频 [3, N, H×8, W×8]

例如：1280×720, 81帧
  编码：[3, 81, 720, 1280] → [16, 21, 90, 160]
  解码：[16, 21, 90, 160]  → [3, 81, 720, 1280]
```

---

## 五、msprof 性能分析指南

### 5.1 Timeline 布局说明

msprof 是昇腾 NPU 的性能分析工具，Timeline 视图用于观察模型执行的时序关系：

- **横轴（X 轴）**：时间线，从左到右表示时间推移，单位为微秒（μs）或毫秒（ms）
- **纵轴（Y 轴）**：不同的执行层级/线程/流（Stream），每一行代表一个独立的执行通道
- **颜色**：不同颜色代表不同类型的操作（如计算、通信、内存拷贝等）
- **层级嵌套**：上层是粗粒度的 nn.Module 调用，展开后可看到细粒度的子操作

### 5.2 关键模块在 Timeline 中的识别

| Timeline 中的名称 | 对应代码位置 | 含义 |
|------------------|------------|------|
| `T5Encoder_0` | `wan/modules/t5.py` | T5 文本编码器，执行文本特征提取 |
| `torch/nn/modules/module.py(1779):_call_impl` | PyTorch 框架层 | nn.Module 的标准调用入口，表示某个模块正在执行 forward |
| `sp_dit_forward` | `wan/distributed/sequence_parallel.py` | 序列并行版 DiT 前向传播（monkey-patch 替换了原始 forward） |
| `WanAttentionBlock_0` ~ `WanAttentionBlock_39` | `wan/modules/model.py` | 40 个注意力块，从 0 编号到 39 |
| `sp_attn_forward` | `wan/distributed/sequence_parallel.py` | 序列并行版自注意力前向传播 |
| `xFuserLongContextAttention` | `wan/modules/attn_layer.py` | Ulysses + Ring Attention 的序列并行注意力实现 |
| `BatchMatMul`（AICore 视图） | 底层算子 | 对应代码中的 nn.Linear 操作 |

### 5.3 如何在 Timeline 中找到 40 个 Block

1. **展开 `sp_dit_forward`**（或 `WanModel.forward`）：这是 DiT 的顶层调用
2. **观察重复模式**：40 个 `WanAttentionBlock_X` 会以相同的结构依次出现
3. **编号从 0 到 39**：`WanAttentionBlock_0` 是第一个 Block，`WanAttentionBlock_39` 是最后一个

### 5.4 Linear 层为什么在 Timeline 中看不到

- `nn.Linear` 在 NPU 上执行时，底层调用的是 **BatchMatMul**（批量矩阵乘法）算子
- 在 Timeline 的 Python 层视图中，Linear 被包含在 Attention 或 FFN 的调用栈中，不会单独显示
- 需要切换到 **AICore 视图** 才能看到 BatchMatMul 算子

### 5.5 xFuserLongContextAttention 的含义

这是序列并行场景下的注意力实现，继承自 `yunchang.LongContextAttention`：

- **Ulysses 并行**：将注意力头（heads）切分到不同卡，通过 All-to-All 通信交换 Q/K/V
- **Ring Attention 并行**：将序列切分到不同卡，通过环形通信传递 K/V
- 在 `sp_attn_forward` 中，WanSelfAttention 的注意力计算被路由到 `xFuserLongContextAttention` 执行

---

## 六、MindIE 优化特性集成方式

MindIE 在原始 Wan2.2 代码基础上，通过以下 6 种方式集成昇腾 NPU 的优化特性：

### 6.1 Monkey-Patch（猴子补丁）

**原理**：在运行时替换类的方法，不修改原始代码文件。

```python
# wan/text2video.py → _configure_model()
if use_sp:
    for block in model.blocks:
        block.self_attn.forward = types.MethodType(
            sp_attn_forward, block.self_attn)  # 替换自注意力 forward
    model.forward = types.MethodType(sp_dit_forward, model)  # 替换 DiT forward
```

**效果**：当启用序列并行时，WanModel.forward → sp_dit_forward，WanSelfAttention.forward → sp_attn_forward，无需修改原始模型代码。

### 6.2 环境变量控制算子选择

**原理**：通过环境变量在运行时选择不同的算子实现。

```python
# wan/modules/model.py
ALGO = int(os.getenv('ALGO', 0))
if ALGO == 1:
    out = attention_forward(q, k, v, opt_type="ascend_laser_attention")  # 高性能 FA
else:
    out = attention_forward(q, k, v, opt_type="fused_attn_score")        # 默认 FA

FAST_LAYERNORM = int(os.getenv('FAST_LAYERNORM', 0))
if FAST_LAYERNORM:
    from mindiesd import fast_layernorm  # 高性能 LayerNorm
```

### 6.3 MindIE-SD 算子库替换

**原理**：MindIE-SD 提供了高性能算子，直接替换 PyTorch 原生实现。

| 优化项 | 原生实现 | MindIE-SD 替换 | 环境变量 |
|--------|---------|---------------|---------|
| Flash Attention | `scaled_dot_product_attention` | `attention_forward(opt_type="fused_attn_score")` | `ALGO=0` |
| 高性能 FA | - | `attention_forward(opt_type="ascend_laser_attention")` | `ALGO=1` |
| LayerNorm | `nn.LayerNorm` | `fast_layernorm` | `FAST_LAYERNORM=1` |
| AdaLayerNorm | 手动 scale+shift | `layernorm_scale_shift` | `ADA_LAYERNORM=1/2` |
| RoPE | 手动旋转 | `rotary_position_embedding(fused=True)` | 默认启用 |
| RMSNorm | 手动计算 | `npu_rms_norm` | 默认启用 |

### 6.4 分布式并行策略

**原理**：通过并行管理器配置不同的并行策略，将计算分配到多张 NPU。

```python
# generate.py
parallel_config = ParallelConfig(
    sp_degree=sp_degree,          # 序列并行度 = ulysses_size × ring_size
    ulysses_degree=args.ulysses_size,
    ring_degree=args.ring_size,
    tp_degree=args.tp_size,
    use_cfg_parallel=(args.cfg_size==2),
    world_size=world_size,
)
init_parallel_env(parallel_config)
```

### 6.5 FSDP 模型分片

**原理**：将模型参数分片存储到多张卡，计算时 All-Gather 收集所需参数。

```python
# wan/distributed/fsdp.py
shard_fn = partial(shard_model, device_id=device_id)

# T5 FSDP
self.text_encoder = T5EncoderModel(..., shard_fn=shard_fn if t5_fsdp else None)

# DiT FSDP
if dit_fsdp:
    model = shard_fn(model)
```

### 6.6 编译优化（ACLGraph）

**原理**：将计算图编译为昇腾 CANN 的 ACLGraph，减少算子下发开销。

```python
# generate.py → add_compilation_args()
--backend_mode off        # 关闭编译
--backend_mode default    # 仅模式融合
--backend_mode aclgraph   # ACLGraph 捕获/重放
--backend_mode aclgraph_with_compile  # 模式融合 + ACLGraph
```

---

## 七、单卡性能测试参数详解

### 7.1 环境变量参数

| 参数 | 取值 | 含义 | 使用建议 |
|------|------|------|---------|
| `ALGO` | 0 / 1 / 3 | Flash Attention 算子选择。**0**=默认 FA 算子（fused_attn_score）；**1**=高性能 FA 算子（ascend_laser_attention）；**3**=量化 FA | A5 设备用 0，A2/A3 设备用 1 |
| `FAST_LAYERNORM` | 0 / 1 | LayerNorm 计算方式。**0**=使用 PyTorch 原生 LayerNorm；**1**=使用 MindIE 高性能 layernorm 算子 | 建议设为 1 提升性能 |

**设置方式：**

```bash
export ALGO=1
export FAST_LAYERNORM=1
```

### 7.2 命令行参数

| 参数 | 含义 | 示例值 | 详细说明 |
|------|------|--------|---------|
| `--task` | 任务类型 | `t2v-A14B` | 指定模型和任务。可选：`t2v-A14B`（文生视频14B）、`i2v-A14B`（图生视频14B）、`ti2v-5B`（文/图生视频5B） |
| `--ckpt_dir` | 模型权重路径 | `/data/weights` | 必须指向包含 T5、VAE、DiT 权重的目录 |
| `--size` | 视频分辨率 | `1280*720` | T2V 支持：720×1280, 1280×720, 480×832, 832×480, 432×768, 768×432 |
| `--frame_num` | 生成帧数 | `81` | 必须满足 4n+1（如 5, 9, 13, ..., 81）。帧数越多，视频越长，计算量越大 |
| `--sample_steps` | 推理步数 | `40` | 扩散去噪的步数。步数越多质量越高但越慢。A14B 默认 40 步 |
| `--sample_shift` | 采样偏移因子 | `12.0` | 影响 noise schedule 的分布。A14B 默认 12.0 |
| `--sample_guide_scale` | CFG 引导强度 | `3.0` 或 `(3.0,4.0)` | 控制生成内容与提示词的对齐程度。值越大越遵循提示词，但可能降低多样性。A14B 默认低噪声 3.0、高噪声 4.0 |
| `--prompt` | 文本提示词 | `"A cat playing..."` | 描述想要生成的视频内容 |
| `--offload_model` | 是否开启 CPU offload | `True` / `False` | 显存不足时开启，将不活跃的模型搬到 CPU，用时间换空间。单卡默认 True，多卡默认 False |
| `--base_seed` | 随机种子 | `0` | 固定种子可复现结果。设为 -1 使用随机种子 |
| `--sample_solver` | 采样求解器 | `unipc` / `dpm++` | UniPC 或 DPM++ 求解器。默认 unipc |

### 7.3 单卡测试命令示例

```bash
export ALGO=1
export FAST_LAYERNORM=1

python generate.py \
    --task t2v-A14B \
    --ckpt_dir /path/to/weights \
    --size 1280*720 \
    --frame_num 81 \
    --sample_steps 40 \
    --prompt "A beautiful sunset over the ocean" \
    --base_seed 0
```

---

## 八、八卡性能测试参数详解

八卡测试在单卡基础上增加了**分布式环境变量**和**并行策略参数**。

### 8.1 额外环境变量

| 参数 | 含义 | 说明 |
|------|------|------|
| `PYTORCH_NPU_ALLOC_CONF` | NPU 显存分配策略 | `expandable_segments:True` 启用可扩展段分配，减少显存碎片化，提高显存利用率 |
| `TASK_QUEUE_ENABLE` | 任务队列开关 | 设为 `2` 启用异步任务队列，提升算子下发和调度的效率 |
| `CPU_AFFINITY_CONF` | CPU 亲和性配置 | 设为 `1` 绑定进程到特定 CPU 核，减少 CPU 上下文切换开销 |
| `TOKENIZERS_PARALLELISM` | Tokenizer 并行 | 设为 `false` 避免多进程场景下 Tokenizer 的死锁问题 |

**设置方式：**

```bash
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
```

### 8.2 分布式启动命令

```bash
torchrun --nproc_per_node=8 --master_port=23459 generate.py ...
```

- `--nproc_per_node=8`：启动 8 个进程，对应 8 张 NPU
- `--master_port=23459`：进程间通信端口

### 8.3 分布式并行参数

| 参数 | 含义 | 默认值 | 说明 |
|------|------|--------|------|
| `--dit_fsdp` | 对 DiT 启用 FSDP | False | 将 DiT 模型参数分片到多卡，减少单卡显存占用 |
| `--t5_fsdp` | 对 T5 启用 FSDP | False | 将 T5 编码器参数分片到多卡 |
| `--cfg_size` | CFG 并行度 | 1 | 设为 2 表示用 2 张卡分别处理条件/无条件分支 |
| `--ulysses_size` | Ulysses 序列并行度 | 1 | 将注意力头切分到多卡，通过 All-to-All 通信 |
| `--ring_size` | Ring Attention 并行度 | 1 | 将序列切分到多卡，通过环形通信传递 K/V |
| `--tp_size` | 张量并行度 | 1 | 张量并行（当前不支持） |
| `--vae_parallel` | 启用 VAE 并行 | False | VAE 解码阶段将 patch 分配到不同卡处理 |

### 8.4 并行度约束公式

**核心公式：**

```
world_size = cfg_size × ulysses_size × ring_size × tp_size
```

对于 8 卡测试（`--cfg_size 2 --ulysses_size 4`）：

```
8 = 2 (CFG) × 4 (Ulysses) × 1 (Ring) × 1 (TP)  ✅
```

> ⚠️ **必须满足**：`cfg_size × ulysses_size × ring_size × tp_size == world_size`，否则程序报错退出。

**最优并行配置（OPTIMAL_PARALLEL）：**

```python
OPTIMAL_PARALLEL = [
    (2,1,1), (2,2,1), (2,4,1), (2,8,1),     # cfg=2 的组合
    (1,2,1), (1,4,1), (1,8,1),               # cfg=1 的组合
    (1,4,2), (1,8,2),                         # 含 ring attention 的组合
]
# 格式：(cfg_size, ulysses_size, ring_size)
```

### 8.5 各并行策略详解

#### CFG 并行（Classifier-Free Guidance Parallelism）

```
┌──────────┐     ┌──────────┐
│  GPU 0   │     │  GPU 1   │
│ 条件预测  │     │ 无条件预测 │
│ context  │     │ context_ │
│ =prompt  │     │ null     │
└────┬─────┘     └────┬─────┘
     │                │
     └──── AllGather ─┘
              │
     noise_pred = uncond + scale × (cond - uncond)
```

- `cfg_size=2`：2 张卡分别独立计算条件/无条件预测
- **无需通信**（除最后的 AllGather），完全独立
- 适合 `guide_scale > 1` 的场景
- 代码路径：`get_classifier_free_guidance_world_size() == 2` → 使用 `arg_all` + `all_gather`

#### Ulysses 序列并行

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  GPU 0   │  │  GPU 1   │  │  GPU 2   │  │  GPU 3   │
│ Heads    │  │ Heads    │  │ Heads    │  │ Heads    │
│ 0-9      │  │ 10-19    │  │ 20-29    │  │ 30-39    │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
     ↕ All-to-All 通信：交换 Q/K/V 的 head 和 sequence 维度
```

- `ulysses_size=4`：40 个注意力头分为 4 组，每组 10 个头
- 通过 All-to-All 通信在 head 维度和 sequence 维度之间转换
- 要求 `num_heads % ulysses_size == 0`（40 % 4 = 0 ✅）
- 代码位置：`xFuserLongContextAttention` → `ulysses_pg` All-to-All

#### Ring Attention 并行

- `ring_size=2`：将序列切分为 2 段，通过环形通信传递 K/V
- 适合超长序列场景
- 代码位置：`xFuserLongContextAttention` → `ring_pg` 环形通信

#### FSDP（Fully Sharded Data Parallel）

```
┌──────────────────────────────────────────┐
│ 完整模型参数                               │
├──────────┬──────────┬──────────┬─────────┤
│  GPU 0   │  GPU 1   │  GPU 2   │  GPU 3  │
│ Shard 0  │  Shard 1 │  Shard 2 │ Shard 3 │
└──────────┴──────────┴──────────┴─────────┘
计算时：All-Gather 收集所需参数 → 计算 → 释放
```

- `--dit_fsdp`：对 DiT 模型使用 FSDP 分片
- `--t5_fsdp`：对 T5 编码器使用 FSDP 分片
- 计算前 All-Gather 收集当前层参数，计算后释放
- 显著减少单卡显存占用，但增加通信开销
- 代码位置：`wan/distributed/fsdp.py` → `shard_model()`

#### VAE 并行

- `--vae_parallel`：VAE 解码阶段并行处理
- 8 卡时：4 卡做 patch 并行 + 2 卡做 pipeline 并行
- 少于 8 卡时：所有卡做 patch 并行
- 代码位置：`wan/vae_patch_parallel.py` → `set_vae_patch_parallel()`

### 8.6 八卡测试完整命令

```bash
export ALGO=1
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export FAST_LAYERNORM=1

torchrun --nproc_per_node=8 --master_port=23459 generate.py \
    --task t2v-A14B \
    --ckpt_dir ${model_base} \
    --size 1280*720 \
    --frame_num 81 \
    --sample_steps 40 \
    --dit_fsdp \
    --t5_fsdp \
    --cfg_size 2 \
    --ulysses_size 4 \
    --vae_parallel \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --base_seed 0
```

**参数分解：**

| 参数 | 值 | 作用 |
|------|-----|------|
| `--cfg_size 2` | 2 | 2 张卡做 CFG 并行（条件/无条件各 1 卡） |
| `--ulysses_size 4` | 4 | 4 张卡做 Ulysses 序列并行（40 头分 4 组，每组 10 头） |
| `--ring_size 1` | 1（默认） | 不使用 Ring Attention |
| `--tp_size 1` | 1（默认） | 不使用张量并行 |
| `--dit_fsdp` | 开启 | DiT 参数分片存储 |
| `--t5_fsdp` | 开启 | T5 参数分片存储 |
| `--vae_parallel` | 开启 | VAE 解码并行 |

**验证：** `2 (CFG) × 4 (Ulysses) × 1 (Ring) × 1 (TP) = 8 = world_size` ✅

---

## 九、高级优化特性

### 9.1 Rainfusion（稀疏注意力）

通过跳过部分注意力计算来加速推理，适用于高噪声时间步之后的阶段。

| 版本 | 说明 | 适用设备 |
|------|------|---------|
| v1 | 空间稀疏注意力（已弃用） | - |
| v2 | 块稀疏注意力 | 通用 |
| v3 | 块稀疏注意力 + mask 缓存 + 首帧保护 | 仅 950 芯片 |

```bash
--use_rainfusion              # 启用稀疏注意力
--sparsity 0.64               # 稀疏度（越大越快，精度越低）
--sparse_start_step -1        # 开始稀疏的步数（-1 自动设为 1/4 总步数）
--rainfusion_type v2          # 选择版本
--mask_refresh_interval 1     # v3 专用：mask 刷新间隔
```

### 9.2 MagCache（注意力缓存加速）

通过缓存注意力结果，跳过冗余计算步骤来加速推理。

```bash
--use_magcache                # 启用 MagCache
--magcache_thresh 0.04        # 累积误差上界
--retention_ratio 0.2         # 不变步的保留比例
--magcache_K 2                # 最大跳步数
--magcache_calibration        # 校准平均幅度比
```

### 9.3 Attention Cache

缓存注意力计算结果，在特定步数范围内复用。

```bash
--use_attentioncache          # 启用 Attention Cache
--attentioncache_ratio 1.2    # 缓存比率
--attentioncache_interval 4   # 缓存间隔
--start_step 12               # 开始步数
--end_step 37                 # 结束步数
```

### 9.4 量化（Quantization）

支持 DiT 模型的 INT8 量化，减少显存占用和计算量。

```bash
--quant_dit_path /path/to/quant  # 指定量化配置路径
```

### 9.5 编译优化

```bash
--backend_mode off                       # 关闭编译
--backend_mode default                   # 仅模式融合
--backend_mode aclgraph                  # ACLGraph 捕获/重放
--backend_mode aclgraph_with_compile     # 模式融合 + ACLGraph
```

> ⚠️ 编译优化仅支持 `ti2v` 任务。

---

## 十、注意事项与常见问题

### 10.1 并行配置约束

- **必须满足**：`cfg_size × ulysses_size × ring_size × tp_size == world_size`
- **Ulysses 约束**：`num_heads % ulysses_size == 0`（A14B 有 40 个头）
- **单卡不支持**：`t5_fsdp`、`dit_fsdp`、`cfg_size>1`、`ulysses_size>1`、`vae_parallel`
- **TP 未实现**：`tp_size > 1` 会抛出 `NotImplementedError`
- **VAE 并行**：卡数必须是 2 的幂

### 10.2 兼容性问题

- `--use_rainfusion` 使用 Python 层面的 `t_idx` 比较，与编译模式（`--backend_mode aclgraph`）不兼容
- `ALGO` 参数需根据设备类型选择：A5 用 0，A2/A3 用 1
- Rainfusion v3 仅支持昇腾 950 芯片
- Attention Cache 仅支持 `ti2v` 任务

### 10.3 性能调优建议

1. **显存不足**：
   - 开启 `--offload_model`（单卡默认开启）
   - 增加 `--dit_fsdp` 和 `--t5_fsdp`
   - 减小 `--frame_num` 或 `--size`

2. **速度优先**：
   - 设置 `ALGO=1`、`FAST_LAYERNORM=1`
   - 设置 `TASK_QUEUE_ENABLE=2`、`CPU_AFFINITY_CONF=1`
   - 启用 `--use_rainfusion` 或 `--use_magcache`

3. **多卡扩展优先级**：
   - 优先增加 `cfg_size`（无额外通信开销）
   - 其次增加 `ulysses_size`（All-to-All 通信）
   - 最后考虑 `ring_size`（环形通信开销较大）

4. **长视频生成**：
   - 增大 `--frame_num`，配合序列并行减少单卡压力
   - 帧数必须满足 `4n+1` 的约束

---

## 十一、关键代码路径速查

| 功能 | 代码位置 | 关键函数/类 |
|------|---------|-----------|
| 程序入口 | `generate.py` | `generate()` |
| 参数解析 | `generate.py` | `_parse_args()` / `_validate_args()` |
| 分布式初始化 | `generate.py` | `dist.init_process_group(backend="hccl")` |
| 并行组配置 | `generate.py` | `init_parallel_env(parallel_config)` |
| T2V 管线创建 | `generate.py` | `wan.WanT2V(config, checkpoint_dir, ...)` |
| T5 编码器加载 | `wan/text2video.py` | `T5EncoderModel(...)` |
| VAE 加载 | `wan/text2video.py` | `Wan2_1_VAE(...)` |
| DiT 模型加载 | `wan/text2video.py` | `WanModel.from_pretrained(...)` |
| 序列并行 patch | `wan/text2video.py` | `_configure_model()` → monkey-patch |
| FSDP 分片 | `wan/text2video.py` | `shard_fn(model)` |
| 视频生成主循环 | `wan/text2video.py` | `WanT2V.generate()` |
| T5 文本编码 | `wan/text2video.py` | `self.text_encoder([prompt], device)` |
| 扩散去噪循环 | `wan/text2video.py` | `for t_idx, t in enumerate(timesteps)` |
| CFG 计算 | `wan/text2video.py` | `noise_pred = uncond + scale × (cond - uncond)` |
| 模型选择 | `wan/text2video.py` | `_prepare_model_for_timestep()` |
| VAE 解码 | `wan/text2video.py` | `self.vae.decode(x0)` |
| DiT 前向传播 | `wan/modules/model.py` | `WanModel.forward()` |
| 序列并行前向 | `wan/distributed/sequence_parallel.py` | `sp_dit_forward()` |
| 注意力前向 | `wan/modules/model.py` | `WanSelfAttention.forward()` |
| 序列并行注意力 | `wan/distributed/sequence_parallel.py` | `sp_attn_forward()` |
| Ulysses+Ring Attention | `wan/modules/attn_layer.py` | `xFuserLongContextAttention` |
| 并行组管理 | `wan/distributed/parallel_mgr.py` | `ParallelConfig` / `init_parallel_env()` |
| FSDP 实现 | `wan/distributed/fsdp.py` | `shard_model()` |
| VAE 并行 | `wan/vae_patch_parallel.py` | `set_vae_patch_parallel()` |
| 模型配置 | `wan/configs/wan_t2v_A14B.py` | `t2v_A14B` EasyDict |
| 共享配置 | `wan/configs/shared_config.py` | `wan_shared_cfg` EasyDict |
