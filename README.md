# BrainToken

BrainToken is a monorepo for converting MEG/EEG HDF5 recordings into validated NeuroToken datasets through a LangGraph backend and a Tauri/React desktop client.

## Repository layout

- `agent/`: LangGraph Agent Server, source adapters, deterministic pipeline stages, and Python tests.
- `neurotoken-desktop/`: NeuroToken Desktop client built with React, TypeScript, and Tauri.

Generated datasets, model checkpoints, local input data, runtime state, credentials, and build dependencies are intentionally excluded from Git.

## Backend setup

```bash
cd agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[server,test]"
cp .env.example .env
./run_agent_server.sh
```

Configure local data roots, output roots, the scientific Python executable, model paths, and API credentials in `agent/.env` and your local pipeline configuration. Do not commit those values.

## Desktop setup

```bash
cd neurotoken-desktop
npm install
npm run tauri:dev
```

The backend listens on `http://127.0.0.1:2024` by default.

## Validation

Backend:

```bash
cd agent
pytest
```

Desktop:

```bash
cd neurotoken-desktop
npm run test
npm run typecheck
npm run lint
```

See `agent/agent_pipeline/README.md` and `neurotoken-desktop/README.md` for component-specific details.
