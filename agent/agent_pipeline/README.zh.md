# NeuroToken Agent Pipeline 中文说明

这是一个供 NeuroToken Desktop 使用的 LangGraph Agent Server，用于将 MEG/EEG HDF5 数据转换为经过验证的 NeuroToken 数据集。

## 架构

`../neurotoken-desktop` 中的 Tauri/React 客户端连接 `neurotoken_agent`：

```text
NeuroToken Desktop
  -> LangGraph Agent Server
    -> 计划审批
    -> 创建隔离 workflow/run
    -> 执行并审阅阶段
      -> preflight -> package -> tokenize -> readme -> docs -> validate -> reconstruct
```

LLM 只负责意图理解、结构化计划和结果解释；真正的数据处理由确定性 Python 工具完成。

项目不再提供面向用户的 Python CLI。请先启动后端，再从 monorepo 根目录启动 Desktop：

```bash
cd agent
./run_agent_server.sh

cd ../neurotoken-desktop
npm install
npm run tauri:dev
```

后端默认监听 `http://127.0.0.1:2024`。

## 环境安装

```bash
cd agent
conda create -n neurotoken-server -c conda-forge python=3.11
conda run -n neurotoken-server python -m pip install -e ".[server,test]"
cp .env.example .env

cd ../neurotoken-desktop
npm install
```

在 `.env` 中配置：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `NEUROTOKEN_MODEL`
- `NEUROTOKEN_ALLOWED_INPUT_ROOTS`
- `NEUROTOKEN_ALLOWED_OUTPUT_ROOTS`
- `NEUROTOKEN_PIPELINE_PYTHON`
- 启用 adapter 编写时使用独立的 `CURSOR_API_KEY`

## Pipeline 阶段

- `preflight`：检查配置和源 HDF5 结构。
- `package`：通过 adapter 统一源数据，写出 WebDataset tar 和 sidecar。
- `tokenize`：将连续神经信号编码成 token ID。
- `readme`：生成数据集 README。
- `docs`：生成数据字典和 JSON Schema。
- `validate`：验证发布目录与 metadata。
- `reconstruct`：从 token 解码信号并计算重建指标。

执行前必须在 NeuroToken Desktop 审批卡中确认计划。每个阶段完成后，用户可以继续、暂停或取消。

## Agent 内部工具

`preflight`、`readme`、`docs`、`validate` 保持为 Agent 管理的短操作；长时间运行的 `package`、`tokenize`、`reconstruct` 通过 tmux durable job 执行。科学计算 worker 包括：

- `reference_scripts/2_neural2token_human_new.py`
- `reference_scripts/3_make_readme.py`
- `reference_scripts/4_generate_release_docs_and_schemas.py`
- `reference_scripts/5_validate_neurotoken_dataset.py`
- `reference_scripts/6_verify_token_reconstruction.py`

这些脚本保留命令行参数，是 Agent Server 与科学计算环境之间的内部协议，不是用户入口。通过 `NEUROTOKEN_PIPELINE_PYTHON`，它们可以继续运行在包含 PyTorch、UniTok 和 checkpoint 依赖的 `omni` 环境中。

## 隔离运行与恢复

每个批准的计划创建独立目录。`package`、`tokenize`、`reconstruct` 每次执行还会创建一个新的、不可复用的 attempt 目录：

```text
<output-root>/runs/<workflow-id>/<run-id>/
├── run_manifest.json
├── configs/pipeline_config.toml
├── manifests/<stage>.json
├── locks/<stage>.lock
├── attempts/
│   └── <stage>/<stage-execution-id>/
│       ├── job.json
│       ├── status.json
│       ├── stdout.log
│       └── stderr.log
└── <source-slug>/
    ├── tar/
    ├── neurotoken/
    └── reports/
```

`job.json` 保存不可变的启动规格，`status.json` 记录 attempt 生命周期，`stdout.log` 和 `stderr.log` 在 Agent Server 进程之外持久保存 worker 输出。tmux session 使用 `NEUROTOKEN_TMUX_SOCKET` 指定的专用 socket；启动和 reconcile job 的所有进程必须使用同一个 socket 路径。

运行中的 tmux session 可以跨越 Agent Server、Desktop 和 SSH 重启。恢复时，Agent Server 会把持久化的 attempt 状态、该 socket 上的 session 状态和阶段产物进行 reconcile。tmux session 是否存在只提供存活性证据：session 消失不等于失败，session 结束也绝不代表成功。成功的唯一权威是状态为 completed 的阶段 manifest，以及 checksum 与 manifest 记录一致的 report。

取消只能针对精确匹配的活动 attempt，并持久化取消状态；不得覆盖已经有效的 completed manifest/report。只有 reconcile 确认旧 attempt 已无 worker 运行时才允许 retry。retry 保留旧 attempt，并使用新的 attempt 目录、execution ID 和 tmux session；不得复用日志，也不得把旧 attempt 原地改造成新执行。状态不明确、中断、失败或已取消的 attempt 不会被自动视为成功，也不会被破坏性自动重试。

## Adapter 自动编写

未知 HDF5 可以进入 adapter authoring。Cursor SDK 在隔离 worktree 中生成补丁，只允许修改 `agent_pipeline/adapters/`，并且必须经用户审批后才能应用。补丁记录包含 owner/thread 信息，可在 NeuroToken Desktop 中安全回滚。
