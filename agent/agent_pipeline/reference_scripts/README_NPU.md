# NeuroToken NPU (Ascend) 适配版使用说明

## 概述

本目录包含 NeuroToken 流水线的 **NPU（华为 Ascend）适配版**脚本，可在昇腾 NPU 设备上运行神经数据 tokenization 及验证流程。

NPU 版脚本以 `_npu` 后缀命名，与原始 CUDA 版脚本并存，互不影响。

## 适配文件清单

| NPU 版文件 | 原始文件 | 说明 |
|-----------|---------|------|
| `2_neural2token_human_npu.py` | `2_neural2token_human_new.py` | Stage 2: tar 包 → token ids 转换 |
| `6_verify_token_reconstruction_npu.py` | `6_verify_token_reconstruction.py` | Stage 6: token 重建质量验证 |

其余文件（Stage 0/1/3/4/5、adapters、config、report）不涉及 GPU 操作，无需适配，直接复用即可。

## 前置条件

### 1. 环境依赖

```bash
# 基础依赖
pip install torch torch_npu numpy pandas pyyaml tqdm webdataset h5py tomli

# 可选（Stage 5 验证用）
pip install jsonschema
```

### 2. brainmu-H 项目

确保 `brainmu-H-npu` 目录存在且已完成 NPU 适配：

```bash
ls /home/guojinjie/brainmu-H-npu/modeling/unitok/
# 应包含: unitok.py  vqvae.py  rope.py  quant.py  vitamin.py  discriminator.py
```

### 3. 预训练 checkpoint

```bash
ls /home/guojinjie/brainmu-H-npu/results/tok-0507-16-rope-ft/
# 应包含: generator.pt / ema_generator.pt 等权重文件
```

## 配置文件

### TOML 配置模板（NPU 专用）

复制原始配置并修改路径：

```bash
cp ../configs/nod_eeg_demo.toml ../configs/nod_eeg_demo_npu.toml
```

编辑 `nod_eeg_demo_npu.toml`，修改以下字段：

```toml
[tokenizer]
# 指向 NPU 适配版 brainmu-H 项目
project_root = "/home/guojinjie/brainmu-H-npu"
checkpoint_root = "/home/guojinjie/brainmu-H-npu/results/tok-0507-16-rope-ft"
train_config = "/home/guojinjie/brainmu-H-npu/hdata/configs/tok_human.yaml"

[runtime]
# auto 模式会按 NPU → CUDA → CPU 优先级自动检测
device = "auto"
# NPU 支持 bfloat16 autocast
use_bfloat16 = true
```

## 使用方法

### Stage 2: Tokenization（核心步骤）

```bash
cd /home/guojinjie/tokenization_agent/demo/scripts

# === EEG 数据 tokenization ===
python 2_neural2token_human_npu.py \
  --config ../configs/nod_eeg_demo_npu.toml \
  --modality eeg \
  --overwrite

# === MEG 数据 tokenization ===
python 2_neural2token_human_npu.py \
  --config ../configs/nod_eeg_demo_npu.toml \
  --modality meg \
  --overwrite

# === fMRI 数据 tokenization ===
python 2_neural2token_human_npu.py \
  --config ../configs/nod_eeg_demo_npu.toml \
  --modality fmri \
  --overwrite
```

### 命令行参数完整说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | Path | **必填** | TOML 流水线配置文件 |
| `--input-base` | Path | TOML `paths.tar_root` | 含 session 文件夹 + data.tar 的输入根目录 |
| `--output-base` | Path | TOML `paths.token_root` | 输出 neurotoken_dataset 目录 |
| `--project-root` | Path | TOML `tokenizer.project_root` | brainmu-H-npu 项目根目录 |
| `--checkpoint-root` | Path | TOML `tokenizer.checkpoint_root` | checkpoint 根目录（自动选最新 step） |
| `--checkpoint-path` | Path | 无 | 显式指定 checkpoint（覆盖自动选择） |
| `--train-config` | Path | TOML `tokenizer.train_config` | tok_human.yaml 配置文件 |
| `--modality` | eeg/meg/fmri | TOML `dataset.modality` | 神经信号模态 |
| `--device` | string | auto（NPU→CUDA→CPU） | 加速器设备 |
| `--use-bfloat16` | flag | false | 开启 bfloat16 autocast（NPU/CUDA 均支持） |
| `--no-ema` | flag | false | 使用原始 generator 而非 EMA |
| `--data-seed` | int | 42 | 随机种子 |
| `--storage-dtype` | fp16/fp32 | fp16 | 数据存储精度 |
| `--max-samples` | int | 无 | 每 session 最多处理样本数（烟雾测试用） |
| `--max-sessions` | int | 无 | 最多处理 session 数（烟雾测试用） |
| `--overwrite` | flag | false | 删除输出目录后重新写入 |

### 烟雾测试（快速验证）

```bash
# 仅处理 1 个 session 的前 2 个 chunk
python 2_neural2token_human_npu.py \
  --config ../configs/nod_eeg_demo_npu.toml \
  --max-sessions 1 \
  --max-samples 2 \
  --overwrite
```

### Stage 6: Token 重建验证

```bash
python 6_verify_token_reconstruction_npu.py \
  --config ../configs/nod_eeg_demo_npu.toml \
  --max-sessions 1 \
  --max-samples 5
```

该脚本会：
1. 加载 Stage 2 生成的 `token_ids.npy`
2. 通过 UniTok decoder 将 token ids 解码回原始信号
3. 计算 Pearson / MSE / MAE / SSIM 等重建质量指标
4. 验证 token ids 与 checkpoint 的一致性

## 不使用 TOML 配置的完整命令行示例

如果不想用 TOML，可以直接通过命令行参数覆盖所有配置：

```bash
python 2_neural2token_human_npu.py \
  --config ../configs/nod_eeg_demo_npu.toml \
  --project-root /home/guojinjie/brainmu-H-npu \
  --checkpoint-root /home/guojinjie/brainmu-H-npu/results/tok-0507-16-rope-ft \
  --train-config /home/guojinjie/brainmu-H-npu/hdata/configs/tok_human.yaml \
  --input-base ../outputs/tar_packages_new \
  --output-base ../outputs/neurotoken_dataset \
  --modality eeg \
  --device auto \
  --use-bfloat16 \
  --overwrite
```

## 输出结构

```
neurotoken_dataset/
├── manifest.json              # 数据集清单
├── subjects.json              # 受试者索引
├── tokenizer/
│   ├── codebook.npz           # 码本向量
│   └── codebook_metadata.json # 码本元数据
├── stimuli/                   # 刺激材料（可后续填充）
└── sessions/
    └── <subject>_<session>/
        ├── <modality>_token_ids.npy   # token ID 数组 (total_tokens, num_codebooks)
        ├── token_metadata.json        # token 元数据（含 temporal_index）
        ├── session_metadata.json      # session 元数据
        ├── events_info.json           # 事件信息
        └── <modality>_channels.json   # 通道元数据
```

## NPU 适配要点

### 自动设备检测

设置 `--device auto` 或 `runtime.device = "auto"` 时，检测优先级为：

```
NPU (Ascend) → CUDA (GPU) → CPU
```

### autocast 兼容

CUDA 版硬编码了 `torch.amp.autocast('cuda', ...)`，NPU 版改为动态检测：

```python
# CUDA 版（旧）
torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16)

# NPU 版（新）
torch.amp.autocast(device.type, enabled=True, dtype=torch.bfloat16)
```

### brainmu-H-npu 底层改动

`/home/guojinjie/brainmu-H-npu/` 中已做以下适配：

| 文件 | 改动 |
|------|------|
| `modeling/unitok/unitok.py` | `autocast('cuda')` → `autocast(device.type)`、添加 `import torch_npu` |
| `modeling/unitok/vqvae.py` | `torch.cuda.amp.autocast` → `torch.amp.autocast`、NPU contiguous 适配 |
| `modeling/unitok/rope.py` | NPU 上跳过 fused SDPA |
| `hdata/distributed_iterable_dataset.py` | `.cuda()` → device-aware helper |
| `train/pretrain_tok.py` | 全部 `torch.cuda.*` 统一替换为跨加速器辅助函数 |

## 完整流水线（所有 Stage）

```bash
# Stage 0: 预检
python 0_preflight_check.py --config ../configs/nod_eeg_demo_npu.toml

# Stage 1: 打包 H5 → tar
python 1_make_tar_packages4BrainToken_final.py --config ../configs/nod_eeg_demo_npu.toml

# Stage 2: tar → token ids（NPU 版）
python 2_neural2token_human_npu.py --config ../configs/nod_eeg_demo_npu.toml --overwrite

# Stage 3: 生成 README
python 3_make_readme.py --config ../configs/nod_eeg_demo_npu.toml

# Stage 4: 生成文档和 Schema
python 4_generate_release_docs_and_schemas.py --config ../configs/nod_eeg_demo_npu.toml

# Stage 5: 数据集校验
python 5_validate_neurotoken_dataset.py --config ../configs/nod_eeg_demo_npu.toml

# Stage 6: Token 重建验证（NPU 版）
python 6_verify_token_reconstruction_npu.py --config ../configs/nod_eeg_demo_npu.toml
```

或使用流水线运行器一键执行：

```bash
python run_neurotoken_pipeline.py \
  --config ../configs/nod_eeg_demo_npu.toml \
  --stages preflight package tokenize readme docs validate reconstruct \
  --python $(which python) \
  --smoke-test
```

> **注意**：`run_neurotoken_pipeline.py` 中的 `STAGE_SCRIPTS` 字典（第 24 行）当前引用了 `2_neural2token_human_new.py`。如需在流水线中使用 NPU 版，请将该行改为 `2_neural2token_human_npu.py`。

## 故障排查

### NPU 不可用

```python
# 验证 NPU 是否被检测
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

### 显存不足

- 使用 `--max-samples` / `--max-sessions` 限制处理量
- 关闭 `--use-bfloat16` 降为 fp32
- 减小 `chunking.split_points`

### checkpoint 找不到

```bash
# 检查 checkpoint 目录结构
ls /home/guojinjie/brainmu-H-npu/results/tok-0507-16-rope-ft/
# 应包含 generator.pt 或 ema_generator.pt

# 或显式指定 checkpoint 路径
python 2_neural2token_human_npu.py ... --checkpoint-path /path/to/checkpoint.pt
```
