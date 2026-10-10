# NeuroToken 单数据集 CUDA 并行流水线

本目录提供两个入口：

- `run_neurotoken_pipeline_multi_gpu.py`：推荐使用，完整执行 Stage 0–6；
- `run_single_dataset_multi_gpu.py`：只执行 Stage 1/2，主要用于调试或已有 tar 的 tokenization。

并行设计和代码调用关系见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 1. 功能范围

完整入口的执行顺序是：

```text
Stage 0  preflight
Stage 1  H5 → tar，按 H5 做 CPU 多进程
Stage 2  tar → token，每块 CUDA GPU 一个持久 worker
Stage 3  README
Stage 4  docs/schema
Stage 5  release validation
Stage 6  token reconstruction（当前为单设备）
```

这是单数据集 CUDA 入口，不支持：

- 一次提交多个数据集；
- NPU；
- Stage 6 多 GPU reconstruction。

## 2. 推荐流程

进入脚本目录：

```bash
cd /home/guojinjie/tokenization_agent/demo/scripts
```

### 2.1 先做 dry-run

如果是重跑并准备覆盖旧 tar/token 输出：

```bash
python parallel/run_neurotoken_pipeline_multi_gpu.py \
  --config ../configs/nod_eeg_demo.toml \
  --devices cuda:0 cuda:1 \
  --overwrite-tar-output \
  --overwrite-token-output \
  --dry-run
```

Dry-run 会：

- 打印 Stage 0–6 的实际子命令；
- 检查配置和输出根目录前置条件；
- 检查清理路径是否安全；
- 不启动任何 Stage；
- 不删除任何文件；
- 不覆盖上一次真实运行报告。

输出：

```text
pipeline_dry_run_report.json
pipeline_dry_run_summary.md
```

如果正式运行会因为非空输出目录或不安全路径而失败，dry-run 状态为：

```text
dry_run_blocked
```

### 2.2 正式运行：自动资源选择

```bash
python parallel/run_neurotoken_pipeline_multi_gpu.py \
  --config ../configs/nod_eeg_demo.toml \
  --overwrite-tar-output \
  --overwrite-token-output
```

自动行为：

- H5 worker：根据 CPU affinity、可用 RAM、最大 H5 signal 大小和 H5 数量规划；
- GPU：选择所有满足安全显存预算的可见 CUDA 设备；
- 每块选中 GPU 启动一个模型 worker；
- GPU 数不超过 session 数和 `--max-gpus`。

### 2.3 正式运行：显式资源

```bash
python parallel/run_neurotoken_pipeline_multi_gpu.py \
  --config ../configs/nod_eeg_demo.toml \
  --h5-workers 8 \
  --devices cuda:0 cuda:1 \
  --token-worker-memory-gb 16 \
  --overwrite-tar-output \
  --overwrite-token-output
```

`--devices` 的顺序会保留，重复设备会去重。每块设备仍然只有一个 worker，不存在 `--token-workers` 参数。

## 3. 输出覆盖与安全

### 第一次运行

如果 `paths.tar_root` 和 `paths.token_root` 不存在或为空，可以不传：

```text
--overwrite-tar-output
--overwrite-token-output
```

### 重跑

如果输出目录非空，完整入口默认拒绝继续，避免：

- 已删除 H5 对应的旧 tar session 残留；
- 旧 token session 与本次部分结果混合；
- manifest 与目录内容不一致。

重跑时需要明确授权：

```text
--overwrite-tar-output    删除整个 paths.tar_root
--overwrite-token-output  删除整个 paths.token_root
```

这两个参数会删除配置指定的输出根，请先用 `--dry-run` 核对路径。

完整入口还会拒绝：

- `tar_root` 或 `token_root` 是普通文件；
- 输出路径自身或任意父组件是 symlink；
- 清理目标等于或包含 H5 输入；
- 清理目标等于或包含 tokenizer project、checkpoint 或 train config；
- 清理目标覆盖配置、报告或另一个输出根。

## 4. Smoke test

```bash
python parallel/run_neurotoken_pipeline_multi_gpu.py \
  --config ../configs/nod_eeg_demo.toml \
  --devices cuda:0 cuda:1 \
  --overwrite-tar-output \
  --overwrite-token-output \
  --smoke-test
```

`--smoke-test` 的默认限制是：

| Stage | 限制 |
|---|---|
| Stage 0 | 最多检查 1 个 H5 |
| Stage 1 | **仍打包全部 H5** |
| Stage 2 | 最多 1 个 session，每个 session 最多 2 个 tar sample |
| Stage 3 | 允许关键 README 字段暂时为 placeholder |
| Stage 6 | 最多 1 个 session，每个 session 最多 2 个 source sample |

这里的 sample 是 tar/source chunk，不是时间点，也不一定等于一个模型窗口。较长 sample 仍可能切成多个模型窗口。

Smoke test 用于检查：

- H5、tar、token 格式是否连通；
- checkpoint 和模型能否加载；
- CUDA worker 能否启动；
- metadata、manifest、validation 和 reconstruction 是否能执行。

它产生的是受限测试数据和抽样指标，不应作为正式 release 或最终质量结论。

如果需要自定义限制：

```bash
--max-sessions 2
--max-samples 10
--max-reconstruction-sessions 2
--max-reconstruction-samples 10
```

## 5. 选择部分 Stage

`--stages` 可以执行 canonical 顺序的子集：

```bash
# 只运行输入检查
python parallel/run_neurotoken_pipeline_multi_gpu.py \
  --config ../configs/nod_eeg_demo.toml \
  --stages preflight

# 只运行 Stage 0–2
python parallel/run_neurotoken_pipeline_multi_gpu.py \
  --config ../configs/nod_eeg_demo.toml \
  --stages preflight package tokenize \
  --overwrite-tar-output \
  --overwrite-token-output

# 已有 token release，只运行验证和重建
python parallel/run_neurotoken_pipeline_multi_gpu.py \
  --config ../configs/nod_eeg_demo.toml \
  --stages validate reconstruct
```

允许的顺序固定为：

```text
preflight package tokenize readme docs validate reconstruct
```

重复或乱序会报错，例如：

```text
--stages tokenize package
--stages package package
```

执行子集时，用户负责保证被跳过 Stage 的输出已经存在且与当前配置一致。

## 6. Stage 6 设备

Stage 2 的多 GPU 列表不控制 Stage 6。Stage 6 当前运行 canonical 单设备 reconstruction：

```bash
--reconstruction-device cuda:0
```

如果不指定，使用 TOML：

```toml
[runtime]
device = "cuda:0"
```

可另外设置：

```bash
--max-reconstruction-chunk-batch-size 8
--min-mean-pearson 0.1
```

正式全量 reconstruction 的成本可能较高；建议先用 smoke test 或显式 sample/session 限制确认流程。

## 7. 常用参数

### 完整流水线

| 参数 | 含义 |
|---|---|
| `--config PATH` | TOML 配置 |
| `--report-root PATH` | 覆盖报告目录 |
| `--stages ...` | canonical 顺序的 Stage 子集 |
| `--dry-run` | 只打印、检查，不执行或删除 |
| `--smoke-test` | 使用小规模 Stage 0/2/6 限制 |
| `--python PATH` | 子 Stage 使用的 Python 解释器 |

### Stage 1

| 参数 | 含义 |
|---|---|
| `--h5-workers N` | H5 并发数；省略或 `0` 表示自动 |
| `--overwrite-tar-output` | Stage 1 前安全清理整个 tar_root |

### Stage 2

| 参数 | 含义 |
|---|---|
| `--devices auto` | 自动选择安全 GPU |
| `--devices cuda:0 cuda:1` | 显式设备列表 |
| `--max-gpus N` | auto 模式最多选择 N 块 GPU，默认 64 |
| `--token-worker-memory-gb N` | 每模型副本显存规划估值，默认 16 GiB 或 TOML 配置 |
| `--max-sessions N` | 最多 tokenization N 个 session |
| `--max-samples N` | 每个 session 最多 N 个 tar sample |
| `--overwrite-token-output` | Stage 2 前清理整个 token_root |

### Stage 3–5

| 参数 | 含义 |
|---|---|
| `--allow-placeholder-readme` | README 关键字段为 TODO/placeholder 时仍继续 |
| `--strict-validation` | validation warning 也视为失败 |
| `--check-event-duration-semantics` | 启用可选 event duration 语义检查 |

## 8. 报告文件

真实完整运行在 report root 写入：

```text
pipeline_report.json
pipeline_summary.md
preflight_report.json
stage1_parallel_report.json
stage2_multi_gpu_report.json
validate_report.json
reconstruct_report.json
```

`pipeline_report.json` 包含：

- 每个 Stage 的命令、状态、退出码和耗时；
- 严格屏障停止位置；
- Stage 1/2 子报告；
- H5 worker 与 GPU 资源计划；
- GPU 启动、运行和存活副本信息；
- session 失败来源；
- overwrite 策略和请求设备。

任意 Stage 失败时，程序返回非零退出码。

## 9. 独立运行 Stage 1/2

调试时可以只用两阶段入口。

### H5 → tar 后停止

```bash
python parallel/run_single_dataset_multi_gpu.py \
  --config ../configs/nod_eeg_demo.toml \
  --package-only \
  --h5-workers 4
```

### 使用已有 tar 做多 GPU tokenization

```bash
python parallel/run_single_dataset_multi_gpu.py \
  --config ../configs/nod_eeg_demo.toml \
  --skip-h5-to-tar \
  --devices cuda:0 cuda:1 \
  --overwrite-token-output
```

该低层入口不执行 Stage 0、3、4、5、6，也不提供完整父入口的 tar/token 清理路径保护。正式运行优先使用：

```text
run_neurotoken_pipeline_multi_gpu.py
```

## 10. 常见问题

### `dry_run_blocked`

查看 dry-run summary 中的 Blockers。最常见原因是输出根非空但没有提供对应 overwrite 参数。

### `No CUDA devices were selected`

检查：

```bash
nvidia-smi
```

以及当前进程的 `CUDA_VISIBLE_DEVICES`。也可以显式传入设备，但显存不足时仍可能 OOM。

### 显式设备有显存 warning

`--token-worker-memory-gb` 只是规划估值，不限制 PyTorch 实际分配。只有在实测确认模型峰值后才应降低该值。

### Stage 3 报 placeholder

补充 TOML 中的 dataset name、short description 和 tokenizer checkpoint 描述；只在 smoke/debug 时使用 `--allow-placeholder-readme`。

### Stage 5 或 Stage 6 失败

父流水线会停止并保留对应 JSON report。先处理 `findings` 和 `failed_session_errors`，不要跳过失败继续发布。
