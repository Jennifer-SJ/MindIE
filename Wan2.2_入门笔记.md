# Wan2.2 视频生成模型入门笔记

---

## 一、项目概览

Wan2.2 是一个视频生成模型，支持 **Text-to-Video (T2V)** 和 **Image-to-Video (I2V)** 两种任务。项目位于 `/Users/yujie/MindIE/Wan2.2/`，核心入口文件为 `generate.py`。

---

## 二、核心文件结构

| 文件路径 | 功能说明 |
|---------|---------|
| `generate.py` | 主入口脚本，负责参数解析、模型初始化、视频生成 |
| `wan/modules/model.py` | 定义 WanModel 核心架构（含 40 个 AttentionBlock） |
| `wan/text2video.py` | T2V 管线实现（WanT2V 类） |
| `wan/modules/attn_layer.py` | 注意力层实现，含 xFuserLongContextAttention |
| `wan/distributed/sequence_parallel.py` | 序列并行实现，monkey-patch forward 方法 |
| `wan/distributed/parallel_mgr.py` | 并行组管理（CFG/SP/Ulysses/TP） |
| `wan/distributed/fsdp.py` | FSDP 分布式训练实现 |
| `wan/vae_patch_parallel.py` | VAE 并行实现 |
| `wan/utils/magcache.py` | MagCache 优化实现 |
| `wan/configs/` | 模型配置文件（wan_t2v_A14B.py 等） |

---

## 三、模型架构详解

### 3.1 整体流程

```
用户输入 Prompt
      ↓
  T5 Encoder（文本编码）
      ↓
  DiT Transformer（40 层 WanAttentionBlock）
      ↓
  VAE Decoder（潜空间 → 像素空间）
      ↓
  输出视频
```

### 3.2 WanModel 核心结构

定义在 `wan/modules/model.py` 中：

```python
class WanModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        # ...
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])
```

**关键参数说明：**
- `num_layers=32`：默认 32 层，A14B 模型为 40 层
- `dim=2048`：隐藏层维度
- `ffn_dim=8192`：FFN 中间层维度
- `num_heads=16`：注意力头数
- `patch_size=(1, 2, 2)`：时空 patch 大小

### 3.3 WanAttentionBlock 内部结构

每个 Block 包含：
1. **Self-Attention**：视频 token 之间的自注意力
2. **Cross-Attention**：视频 token 与文本 token 的交叉注意力
3. **FFN**：前馈网络
4. **LayerNorm**：层归一化

---

## 四、msprof 性能分析指南

### 4.1 Timeline 布局说明

- **横轴（X轴）**：时间线，从左到右表示时间推移
- **纵轴（Y轴）**：不同的执行层级/线程/流
- **颜色**：不同颜色代表不同类型的操作

### 4.2 关键模块识别

| Timeline 中的名称 | 对应代码 | 含义 |
|------------------|---------|------|
| `T5Encoder_0` | T5 文本编码器 | 文本特征提取阶段 |
| `sp_dit_forward` | `wan/distributed/sequence_parallel.py` | 序列并行 DiT 前向传播 |
| `WanAttentionBlock_0~39` | `wan/modules/model.py` | 40 个注意力块 |
| `sp_attn_forward` | 序列并行注意力前向 | 注意力计算 |
| `xFuserLongContextAttention` | `wan/modules/attn_layer.py` | 优化的长上下文注意力实现 |

### 4.3 查看方法

1. **全貌视图**：在 msprof 中查看顶层 nn.Module 调用
2. **展开细节**：点击模块展开查看子操作
3. **AICore 视图**：查看底层算子（如 BatchMatMul 对应 Linear 操作）
4. **40 个 Block 的识别**：在 Timeline 中寻找 `WanAttentionBlock_0` 到 `WanAttentionBlock_39` 的重复模式

### 4.4 Linear 层在 msprof 中的表现

- Linear 操作在 AICore 视图中显示为 **BatchMatMul** 算子
- 不会直接显示为 "Linear"，因为底层执行的是矩阵乘法
- 在 `sp_attn_forward` 下看到 `xFuserLongContextAttention` 是正常的，这是优化的注意力实现

---

## 五、MindIE 优化特性集成方式

MindIE 通过以下 6 种方式将 SD 优化特性集成到模型中：

### 5.1 Monkey-Patch（猴子补丁）

替换原有方法实现，不修改原始代码：

```python
# wan/distributed/sequence_parallel.py
# 替换 WanModel.forward 为 sp_dit_forward
WanModel.forward = sp_dit_forward
WanAttentionBlock.forward = sp_attn_forward
```

### 5.2 配置注入

通过配置类传递优化参数：

```python
# wan/configs/ 中的配置文件
# 定义模型结构和优化参数
```

### 5.3 算子替换

使用高性能算子替换原生实现：
- `ALGO=1`：使用高性能 FA 算子
- `FAST_LAYERNORM=1`：使用高性能 LayerNorm 算子

### 5.4 分布式策略

通过并行管理器配置并行策略：

```python
# wan/distributed/parallel_mgr.py
initialize_model_parallel(
    classifier_free_guidance_degree=cfg_size,
    sequence_parallel_degree=sp_degree,
    ulysses_degree=ulysses_size,
    ...
)
```

### 5.5 条件编译/运行时开关

通过环境变量控制优化开关：

```bash
export ALGO=1
export FAST_LAYERNORM=1
```

### 5.6 FSDP 封装

对模型进行 FSDP 包装实现分片：

```python
# wan/distributed/fsdp.py
# 对 T5 和 DiT 分别进行 FSDP 包装
```

---

## 六、单卡性能测试参数详解

### 6.1 环境变量参数

| 参数 | 取值 | 含义 | 使用建议 |
|------|------|------|---------|
| `ALGO` | 0 / 1 | FA 算子选择。0=默认FA算子，1=高性能FA算子 | A5 设备用 0，A2/A3 设备用 1 |
| `FAST_LAYERNORM` | 0 / 1 | LayerNorm 计算方式。0=原生，1=高性能算子 | 建议设为 1 提升性能 |

### 6.2 命令行参数

| 参数 | 含义 | 示例 | 说明 |
|------|------|------|------|
| `--task` | 任务类型 | `t2v-A14B` | 指定模型类型和任务 |
| `--ckpt_dir` | 模型权重路径 | `/path/to/weights` | 必须指向有效的权重目录 |
| `--size` | 视频分辨率 | `1280*720` 或 `832*480` | 支持 (1280,720) 和 (832,480) |
| `--frame_num` | 生成帧数 | `81` | 帧数越多，视频越长，耗时越大 |
| `--sample_steps` | 推理步数 | `40` / `50` | 步数越多质量越高，耗时越大 |
| `--prompt` | 文本提示词 | `"A cat playing piano"` | 描述想要生成的视频内容 |
| `--offload_model` | 是否开启 CPU offload | `True` / `False` | 显存不足时开启，用时间换空间 |
| `--base_seed` | 随机种子 | `0` | 固定种子可复现结果 |

### 6.3 单卡测试命令示例

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

## 七、八卡性能测试参数详解

八卡测试在单卡基础上增加了分布式相关参数。

### 7.1 额外环境变量

| 参数 | 含义 | 说明 |
|------|------|------|
| `PYTORCH_NPU_ALLOC_CONF` | NPU 显存分配策略 | `expandable_segments:True` 启用可扩展段，减少显存碎片 |
| `TASK_QUEUE_ENABLE` | 任务队列开关 | 设为 2 启用异步任务队列，提升调度效率 |
| `CPU_AFFINITY_CONF` | CPU 亲和性配置 | 设为 1 绑定 CPU 核，减少上下文切换 |
| `TOKENIZERS_PARALLELISM` | Tokenizer 并行 | 设为 false 避免多进程死锁 |

### 7.2 分布式命令行参数

| 参数 | 含义 | 说明 |
|------|------|------|
| `--dit_fsdp` | 对 DiT 启用 FSDP | 将 DiT 模型参数分片到多卡，减少单卡显存占用 |
| `--t5_fsdp` | 对 T5 启用 FSDP | 将 T5 编码器参数分片到多卡 |
| `--cfg_size` | CFG 并行度 | 设为 2 表示用 2 张卡分别处理条件/无条件分支 |
| `--ulysses_size` | Ulysses 序列并行度 | 设为 4 表示将序列切分为 4 份分到 4 卡 |
| `--vae_parallel` | 启用 VAE 并行 | VAE 解码阶段并行处理 |

### 7.3 并行度约束

**关键公式：**

```
world_size = cfg_size × ulysses_size × ring_size × tp_size
```

对于 8 卡测试（`--cfg_size 2 --ulysses_size 4`）：

```
8 = 2 (CFG) × 4 (Ulysses) × 1 (Ring) × 1 (TP)
```

> ⚠️ **注意**：cfg_size × ulysses_size 等必须等于 world_size，否则会报错。

### 7.4 各并行策略说明

#### CFG 并行（Classifier-Free Guidance）
- `cfg_size=2`：2 张卡分别计算条件/无条件预测
- 无需通信，完全独立
- 适合 guidance_scale > 1 的场景

#### Ulysses 序列并行
- `ulysses_size=4`：将 token 序列切分为 4 段
- 每张卡处理 1/4 的序列
- 注意力计算后需要 All-Gather 通信

#### FSDP（Fully Sharded Data Parallel）
- 将模型参数分片存储
- 计算时 All-Gather 收集需要的参数
- 计算后释放，减少显存占用
- `--dit_fsdp`：对 DiT 模型使用 FSDP
- `--t5_fsdp`：对 T5 编码器使用 FSDP

#### VAE 并行
- `--vae_parallel`：VAE 解码阶段并行
- 将 VAE 的 patch 分配到不同卡上处理

### 7.5 八卡测试命令示例

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

---

## 八、注意事项与常见问题

### 8.1 并行配置约束

- `cfg_size × ulysses_size × ring_size × tp_size` 必须等于 `world_size`
- `t5_fsdp` 和 `dit_fsdp` 在非分布式环境下不支持
- Tensor Parallel 当前不支持

### 8.2 兼容性问题

- `--use_rainfusion` 使用 Python 层面的 t_idx 比较，与编译模式不兼容
- `ALGO` 参数需根据设备类型选择：A5 用 0，A2/A3 用 1

### 8.3 性能调优建议

1. **显存不足**：开启 `--offload_model` 或增加 FSDP
2. **速度优先**：设置 `ALGO=1`、`FAST_LAYERNORM=1`、`TASK_QUEUE_ENABLE=2`
3. **多卡扩展**：优先增加 `cfg_size`（无通信开销），其次增加 `ulysses_size`
4. **长视频生成**：增大 `--frame_num`，配合序列并行减少单卡压力

---

## 九、关键代码路径速查

| 功能 | 代码位置 |
|------|---------|
| 模型入口 | `generate.py` → `generate()` 函数 |
| 分布式初始化 | `generate.py` → `dist.init_process_group(backend="hccl", ...)` |
| 模型创建 | `wan/text2video.py` → `WanT2V.__init__()` |
| 视频生成 | `wan/text2video.py` → `WanT2V.generate()` |
| DiT 前向传播 | `wan/modules/model.py` → `WanModel.forward()` |
| 序列并行前向 | `wan/distributed/sequence_parallel.py` → `sp_dit_forward()` |
| 注意力计算 | `wan/modules/attn_layer.py` → `xFuserLongContextAttention` |
| 并行组初始化 | `wan/distributed/parallel_mgr.py` → `initialize_model_parallel()` |
| FSDP 封装 | `wan/distributed/fsdp.py` |
| VAE 并行 | `wan/vae_patch_parallel.py` |