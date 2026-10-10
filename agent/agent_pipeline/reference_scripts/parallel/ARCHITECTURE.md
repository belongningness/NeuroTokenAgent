# NeuroToken CUDA 并行架构

本文档解释 `/home/guojinjie/tokenization_agent/demo/scripts/parallel` 的设计思想、进程边界、代码结构，以及它与上级目录中 canonical Stage 0–6 实现的调用关系。命令行使用方法见同目录的 [README.md](README.md)。

## 1. 设计目标与边界

当前实现只面向：

- 单个 TOML 数据集配置；
- 完整 Stage 0–6 流水线；
- Stage 1 按 H5 文件做 CPU 多进程；
- Stage 2 按 session 做多 GPU 调度；
- 每块 CUDA GPU 恰好一个持久 worker 和一个 UniTok 模型副本；
- Stage 之间使用严格成功屏障；
- 输出顺序和最终报告可复现。

## 2. 保留的代码文件

```text
parallel/
├── run_neurotoken_pipeline_multi_gpu.py  # 完整 Stage 0–6 入口
├── run_single_dataset_multi_gpu.py       # 独立 Stage 1/2 入口
├── resource_planner.py                   # CPU、RAM、CUDA 资源规划
├── package_h5_parallel.py                # Stage 1：一个 H5 一个 CPU 任务
├── tokenize_sessions_multi_gpu.py        # Stage 2：中央多 GPU session 调度器
├── stage2_session_worker.py              # Stage 2：单 GPU 持久模型 worker 函数
├── __init__.py
├── ARCHITECTURE.md
└── README.md
```

各文件只承担一个层次的职责：

| 文件 | 职责 | 是否包含 canonical 算法 |
|---|---|---:|
| `run_neurotoken_pipeline_multi_gpu.py` | Stage 0–6 顺序、子进程命令、屏障、总报告、输出目录安全检查 | 否 |
| `run_single_dataset_multi_gpu.py` | 单数据集 Stage 1/2 编排及两阶段报告 | 否 |
| `resource_planner.py` | H5 worker 数量和多 GPU 设备选择 | 否 |
| `package_h5_parallel.py` | H5 任务池及错误汇总 | 否；动态复用 Stage 1 |
| `tokenize_sessions_multi_gpu.py` | 每 GPU 一个 executor、动态 session 派发、worker 故障隔离 | 否 |
| `stage2_session_worker.py` | 模型初始化、dataset prepare、session convert、dataset finalize | 否；动态复用 Stage 2 |

## 3. 总体调用关系

```text
run_neurotoken_pipeline_multi_gpu.py
│
├── Stage 0 子进程
│   └── ../0_preflight_check.py
│
├── Stage 1 子进程
│   └── run_single_dataset_multi_gpu.py --package-only
│       ├── resource_planner.plan_h5_workers()
│       └── package_h5_parallel.run()
│           └── 动态加载 ../1_make_tar_packages4BrainToken_final.py
│
├── Stage 2 子进程
│   └── run_single_dataset_multi_gpu.py --skip-h5-to-tar
│       ├── resource_planner.plan_multi_gpu_devices()
│       └── tokenize_sessions_multi_gpu.process_one_dataset_multi_gpu()
│           ├── 每块 GPU：ProcessPoolExecutor(max_workers=1)
│           └── stage2_session_worker.py
│               └── 动态加载 ../2_neural2token_human_new.py
│
├── Stage 3 子进程：../3_make_readme.py
├── Stage 4 子进程：../4_generate_release_docs_and_schemas.py
├── Stage 5 子进程：../5_validate_neurotoken_dataset.py
└── Stage 6 子进程：../6_verify_token_reconstruction.py
```

完整入口不会把七个 Stage 同时运行。Stage 级别始终串行：

```text
Stage 0 成功
  → Stage 1 全部 H5 成功
    → Stage 2 全部 session 完成并汇总
      → Stage 3
        → Stage 4
          → Stage 5
            → Stage 6
```

任意 Stage 返回非零退出码，父调度器立即停止，不启动后续 Stage，并写入失败的 `pipeline_report.json`。

## 4. Stage 1：H5 级 CPU 并行

### 4.1 调度边界

Stage 1 的最小并行单位是一个 source H5：

```text
H5 列表
├── H5 A → CPU worker 0 → session_A/data.tar
├── H5 B → CPU worker 1 → session_B/data.tar
└── H5 C → CPU worker 2 → session_C/data.tar
```

不同 worker 不共享 adapter、H5 handle 或 session 对象。每个进程重新构造 adapter 和 runtime config，然后独立读取一个 H5 并写入自己的 session 目录。

### 4.2 复用 canonical Stage 1

`package_h5_parallel.py` 通过 `importlib` 动态加载：

```text
../1_make_tar_packages4BrainToken_final.py
```

并直接复用：

- `build_runtime_config()`；
- `package_one_session()`；
- `TAR_FILENAME`。

因此以下行为仍由 canonical Stage 1 决定：

- H5 adapter 解析；
- rest/event chunking；
- downsample；
- event metadata；
- WebDataset tar 成员格式；
- session sidecar。

并行层只替换原来最外层的 H5 串行循环。

### 4.3 Stage 1 屏障

父进程必须收到每个 H5 的结果。任一 H5 失败时，`strict=True` 使 Stage 1 整体失败，Stage 2 不会启动。这样不会从一组不完整或混合的新旧 tar 中生成 token。

## 5. Stage 2：每块 GPU 一个持久 worker

### 5.1 进程所有权

中央协调器为每个选中设备创建独立 executor：

```text
cuda:0 → ProcessPoolExecutor(max_workers=1) → 模型副本 0
cuda:1 → ProcessPoolExecutor(max_workers=1) → 模型副本 1
cuda:N → ProcessPoolExecutor(max_workers=1) → 模型副本 N
```

每个 worker 进程独占：

- 一个 CUDA device；
- 一个 CUDA context；
- 一份 UniTok 模型和 checkpoint；
- 一份 modality transform；
- 当前 session 的临时状态。

没有 DDP、NCCL、参数同步或跨 GPU tensor 传输。GPU 之间只通过父进程分配普通 Python session task。

### 5.2 worker 与 scheduler 分离

`tokenize_sessions_multi_gpu.py` 只负责：

- 创建、探测和关闭 GPU worker；
- 维护 pending session 队列；
- 使用 `FIRST_COMPLETED` 动态补充任务；
- worker crash 后移除对应 GPU；
- 收集并按 `session_order` 排序结果；
- 选择健康 worker 执行最终汇总。

`stage2_session_worker.py` 只负责：

- `initialize_gpu_session_worker()`：加载模型；
- `gpu_worker_info()`：返回启动探测信息；
- `prepare_dataset()`：初始化 token_root、导出 codebook、分配稳定输出名；
- `convert_session()`：转换一个完整 session；
- `finalize_dataset()`：统一写 manifest 和 subjects。

这种分离避免调度器重复实现模型逻辑，也避免模型 worker 参与全局任务分配。

### 5.3 复用 canonical Stage 2

`stage2_session_worker.py` 动态加载：

```text
../2_neural2token_human_new.py
```

模型 worker 调用 canonical Stage 2 提供的：

- project root 和 checkpoint 解析；
- `build_unitok_for_inference()`；
- modality neural transform；
- `convert_session_tar_to_token_ids()`；
- `summarize_session()`；
- codebook export；
- subject index 和相对路径工具。

因此 EEG/MEG/fMRI 预处理、模型窗口切分、micro-batch、event logical token grid 和 token metadata 均保持 canonical 行为。

### 5.4 动态 session 调度

初始时每块健康 GPU 领取一个 session。某个 worker 完成后立即领取下一个 pending session：

```text
时间 ──────────────────────────────────────────────→
cuda:0  session A ─────── session D ── session F
cuda:1  session B ─ session C ──────── session E
```

任务按完成速度动态分配，而不是预先平均切成固定分片。较快 GPU 不需要等待较慢 GPU 的当前 session。

调度单位是完整 session。单个 session 内的 chunk 顺序、token offset 和文件写入仍保持串行；同一 chunk 内可使用 canonical Stage 2 的有界 CUDA window micro-batch。

## 6. 输出隔离与确定性

### 6.1 session 输出隔离

`prepare_dataset()` 在 fan-out 前单点完成：

```text
发现并排序 source session
→ 分配 session_order
→ 生成唯一 output_session_name
→ 确定 output_session_dir
```

每个 worker 只写自己的目录：

```text
token_root/sessions/
├── session_A/
├── session_B/
└── session_C/
```

不同 GPU 不会并发写同一个 token 文件或 `token_metadata.json`。

### 6.2 单点最终写入

所有 session 完成后，一个健康 worker 执行 `finalize_dataset()`，只写一次：

- `manifest.json`；
- `subjects.json`。

最终 session 列表按预先分配的 `session_order` 排序，而不是按 GPU 完成顺序排序。因此相同输入与配置产生稳定的 manifest 顺序。

## 7. 资源规划

`resource_planner.py` 独立于调度代码，分为两类计划。

### 7.1 H5 worker 计划

综合：

- CPU affinity；
- host/cgroup 可用内存；
- 最大 H5 signal 解压后大小；
- H5 文件数量；
- 自动 worker 上限。

默认给操作系统预留 CPU，只使用约 60% 当前可用内存预算，避免进程数只按 CPU 数量设置而造成 OOM。

### 7.2 多 GPU 计划

自动模式对每块可见 CUDA GPU：

1. 读取当前 free/total memory；
2. 预留 `max(2 GiB, total × 10%)`；
3. 默认按每个模型副本 16 GiB 估算；
4. 只选择满足预算的设备；
5. 最终数量不超过 session 数和 `--max-gpus`。

显式设备保持用户顺序并去重。显存不足时保留但告警；设备不可见时拒绝。遥测失败的设备在 auto 模式跳过，在显式模式保留并明确提示无法进行显存安全检查。

资源计划写入 Stage 1/2 JSON report，包括选择结果、检测值、跳过原因和 warning。

## 8. 与原有 pipeline 代码的复用关系

| 原有文件 | 调用方式 | parallel 复用内容 |
|---|---|---|
| `0_preflight_check.py` | 子进程 | 完整 Stage 0 |
| `1_make_tar_packages4BrainToken_final.py` | `importlib` 动态加载 | runtime、单 session 打包和 tar 格式 |
| `2_neural2token_human_new.py` | 每个 GPU worker 动态加载 | 模型构建、预处理、tokenization、metadata、汇总工具 |
| `3_make_readme.py` | 子进程 | 完整 Stage 3 |
| `4_generate_release_docs_and_schemas.py` | 子进程 | 完整 Stage 4 |
| `5_validate_neurotoken_dataset.py` | 子进程 | 完整 Stage 5 |
| `6_verify_token_reconstruction.py` | 子进程 | 完整 Stage 6，当前为单设备 |
| `run_neurotoken_pipeline.py` | Python import | `write_human_summary()` |
| `neurotoken_config.py` | Python import | TOML 加载、字段和路径解析 |
| `neurotoken_report.py` | Python import | 统一 JSON report |
| `adapters/*` | Python import | H5 发现和 session adapter |

parallel 目录不复制 canonical Stage 0–6 算法。原脚本修复格式或模型行为后，并行入口会通过上述调用关系复用更新后的实现。

NPU 脚本不在当前调用图中，也不会被 parallel CUDA 入口加载。

## 9. 输出目录安全策略

完整入口必须防止重跑时把旧 session 混入新 release：

- 非空 `tar_root` 默认拒绝 Stage 1；
- 非空 `token_root` 默认拒绝 Stage 2；
- 只有显式 `--overwrite-tar-output` / `--overwrite-token-output` 才允许清理；
- dry-run 只检查，不删除；
- 普通文件不能作为输出根目录；
- `tar_root` 和 `token_root` 的任何路径组件都不能是 symlink；
- 清理目标不能等于或包含 H5 输入、另一个输出根、配置、报告、tokenizer project、checkpoint 或 train config。

该检查位于完整父入口。Stage 1/2 子入口保留独立运行能力，但正式完整运行应优先使用父入口提供的安全策略。

## 10. 失败模型

| 失败位置 | 行为 |
|---|---|
| Stage 0 | 停止，不启动 Stage 1 |
| 任意 H5 worker | Stage 1 失败，不启动 Stage 2 |
| 单个 session 数据错误 | 记录结构化错误，其余 session 继续 |
| GPU worker crash | 当前 session 失败，移除该 GPU，健康 GPU 继续 |
| 所有 GPU 不可用 | pending session 标记失败，Stage 2 失败 |
| manifest/subjects 汇总失败 | 可尝试另一个健康 worker；全部失败则 Stage 2 失败 |
| Stage 3–6 任意非零退出 | 停止后续 Stage |

Stage 2 即使部分 session 成功，只要存在失败 session，整个 Stage 仍返回失败；父流水线不会继续生成一个看似完整的 release。

## 11. 已移除的旧路径

当前设计只保留单数据集 CUDA 完整入口，因此以下旧实现已移除：

- 单设备、同卡多模型副本入口；
- 多数据集并行入口；
- 旧 accelerator owner；
- 对应的缓存文件。

如果未来恢复多数据集或 NPU，应该新建独立设计，不应把不同设备所有权模型重新混入当前“每块 CUDA GPU 一个 worker”的调用链。
