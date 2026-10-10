"""Cursor SDK coding provider."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

from agent_pipeline.adapter_authoring.models import CodingAgentResult, CodingTask
from agent_pipeline.coding.worktree import changed_allowed, diff_worktree


_RUNNER = r"""
import fs from "node:fs";

const { Agent, configureCursorSdk } = await import(process.env.CURSOR_SDK_JS);
const request = JSON.parse(fs.readFileSync(0, "utf8"));
configureCursorSdk({ local: { useHttp1ForAgent: true } });
const agent = await Agent.create({
  model: { id: request.model },
  apiKey: process.env.CURSOR_API_KEY,
  local: { cwd: request.cwd },
});
try {
  const result = await (await agent.send(request.instruction)).wait();
  console.log(JSON.stringify(result));
} finally {
  await agent[Symbol.asyncDispose]();
}
"""


def sdk_paths() -> tuple[Path, Path]:
    sdk_python = os.environ.get("NEUROTOKEN_CURSOR_PYTHON")
    if sdk_python:
        process = subprocess.run(
            [sdk_python, "-c", "import importlib.util; s=importlib.util.find_spec('cursor_sdk'); print(next(iter(s.submodule_search_locations)))"],
            check=True,
            capture_output=True,
            text=True,
        )
        root = Path(process.stdout.strip()) / "_vendor" / "bridge"
        return root / "bin" / "node", root / "node_modules" / "@cursor" / "sdk" / "dist" / "esm" / "index.js"
    spec = importlib.util.find_spec("cursor_sdk")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cursor-sdk is not installed.")
    root = Path(next(iter(spec.submodule_search_locations))) / "_vendor" / "bridge"
    return root / "bin" / "node", root / "node_modules" / "@cursor" / "sdk" / "dist" / "esm" / "index.js"


class CursorSdkProvider:
    def __init__(self, model: str, cursor_api_key: str | None = None, proxy: str | None = None):
        self.model = model
        self.api_key = cursor_api_key or os.environ.get("CURSOR_API_KEY")
        self.proxy = proxy
        self.timeout = int(os.environ.get("NEUROTOKEN_CURSOR_TIMEOUT_SECONDS", "300"))
        if not self.api_key:
            raise RuntimeError("Cursor SDK requires CURSOR_API_KEY or --cursor-api-key.")
        if self.api_key.startswith("sk-"):
            raise RuntimeError("CURSOR_API_KEY contains the main LLM key; use a separate Cursor key starting with crsr_.")

    def run(self, task: CodingTask) -> CodingAgentResult:
        node, sdk_js = sdk_paths()
        env = os.environ.copy()
        env["CURSOR_API_KEY"] = self.api_key
        env["CURSOR_SDK_JS"] = str(sdk_js)
        if self.proxy:
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
                env[key] = self.proxy
        env["NO_PROXY"] = "127.0.0.1,localhost,::1"
        env["no_proxy"] = "127.0.0.1,localhost,::1"
        env["NODE_USE_ENV_PROXY"] = "1"

        request = json.dumps({"model": self.model, "cwd": str(task.worktree), "instruction": task.instruction})
        try:
            process = subprocess.run(
                [str(node), "--input-type=module", "-e", _RUNNER],
                input=request,
                text=True,
                capture_output=True,
                env=env,
                cwd=task.worktree,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Cursor SDK timed out after {self.timeout} seconds.") from exc
        if process.returncode != 0:
            raise RuntimeError(process.stderr.strip() or process.stdout.strip())

        payload = json.loads(process.stdout)
        if payload["status"] != "finished":
            error = payload.get("error") or payload.get("result") or payload["status"]
            raise RuntimeError(str(error))
        diff = diff_worktree(task.worktree, task.allowed_paths)
        if not changed_allowed(diff, task.allowed_paths):
            raise RuntimeError("Coding agent modified files outside allowed paths.")
        return CodingAgentResult(summary=str(payload.get("result", "")), diff=diff)
