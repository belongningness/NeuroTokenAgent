# NeuroToken Agent Pipeline

Web-first LangGraph Agent Server for converting MEG/EEG HDF5 data into validated NeuroToken datasets.

For Chinese documentation, see `README.zh.md`.

## Architecture

The supported user interface is the Tauri/React app in `../neurotoken-desktop`, connected to the LangGraph graph `neurotoken_agent`.

```text
NeuroToken Desktop
  -> LangGraph Agent Server
    -> plan approval
    -> isolated workflow/run
    -> stage execution and review
      -> preflight -> package -> tokenize -> readme -> docs -> validate -> reconstruct
```

The LLM classifies intent, builds structured plans, and explains results. Deterministic Python tools perform all data processing.

There is no supported end-user Python CLI. Start the backend, then run the Desktop app from the monorepo root:

```bash
cd agent
./run_agent_server.sh

cd ../neurotoken-desktop
npm install
npm run tauri:dev
```

The backend defaults to `http://127.0.0.1:2024`.

## Setup

```bash
cd agent
conda create -n neurotoken-server -c conda-forge python=3.11
conda run -n neurotoken-server python -m pip install -e ".[server,test]"
cp .env.example .env

cd ../neurotoken-desktop
npm install
```

Configure at least the LLM settings, allowed input/output roots, and the scientific worker Python in `.env`:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `NEUROTOKEN_MODEL`
- `NEUROTOKEN_ALLOWED_INPUT_ROOTS`
- `NEUROTOKEN_ALLOWED_OUTPUT_ROOTS`
- `NEUROTOKEN_PIPELINE_PYTHON`
- `CURSOR_API_KEY` when adapter authoring is enabled

`CURSOR_API_KEY` is separate from `OPENAI_API_KEY`.

## Workflow

- `preflight`: inspect configuration and source HDF5 structure.
- `package`: normalize source data through an adapter and write WebDataset tar files and sidecars.
- `tokenize`: encode continuous neural signals into token IDs.
- `readme`: generate the dataset README.
- `docs`: generate the data dictionary and JSON Schemas.
- `validate`: validate the release structure and metadata.
- `reconstruct`: decode tokens and compute reconstruction metrics.

Plans require approval in NeuroToken Desktop before execution. Every completed stage produces a review interrupt where the user can continue, pause, or cancel.

## Agent tools

`preflight`, `readme`, `docs`, and `validate` remain short Agent-owned operations. The long-running `package`, `tokenize`, and `reconstruct` stages run as durable tmux jobs. The scientific workers include:

- `reference_scripts/2_neural2token_human_new.py`
- `reference_scripts/3_make_readme.py`
- `reference_scripts/4_generate_release_docs_and_schemas.py`
- `reference_scripts/5_validate_neurotoken_dataset.py`
- `reference_scripts/6_verify_token_reconstruction.py`

These scripts retain command-line parsers as the private protocol between the Agent Server and the scientific environment. They are not supported user entrypoints. `NEUROTOKEN_PIPELINE_PYTHON` allows them to run in the existing `omni` environment with PyTorch, UniTok, and checkpoint dependencies.

## Durable runs

Each approved plan creates an isolated directory. Every execution of `package`, `tokenize`, or `reconstruct` also gets a new, immutable attempt directory:

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

`job.json` is the immutable launch specification. `status.json` records the attempt lifecycle, while `stdout.log` and `stderr.log` retain worker output outside the Agent Server process. The tmux session uses the dedicated socket configured by `NEUROTOKEN_TMUX_SOCKET`; every process that launches or reconciles jobs must use the same socket path.

A running tmux session survives Agent Server, Desktop, and SSH restarts. On recovery, the Agent Server reconciles persisted attempt state with the session on that socket and the stage artifacts. Session existence is only liveness evidence: a missing session is not failure, and a finished session is never proof of success. The success authority is a completed stage manifest plus its report with a matching recorded checksum.

Cancellation targets only the exact active attempt, records cancellation state, and must not overwrite a valid completed manifest/report pair. Retry is allowed only after reconciliation proves that no worker for the previous attempt is still running. A retry preserves the old attempt and starts a fresh attempt directory, execution ID, and tmux session; it never reuses logs or mutates the previous attempt into a new run. Ambiguous, interrupted, failed, or cancelled attempts are not automatically treated as successful or destructively retried.

## Adapter authoring

Unknown HDF5 layouts can enter the adapter-authoring workflow. Cursor SDK generates a patch in an isolated worktree, only changes `agent_pipeline/adapters/`, and requires explicit review before application. Applied patches retain owner/thread metadata and can be rolled back from NeuroToken Desktop.
