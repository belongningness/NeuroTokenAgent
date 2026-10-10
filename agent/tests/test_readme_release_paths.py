from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).parents[1] / "agent_pipeline" / "reference_scripts" / "3_make_readme.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("neurotoken_stage3", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
stage3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage3)


def test_quick_start_does_not_publish_absolute_notebook_path():
    rendered = stage3._render_quick_start({
        "paths": {"notebook_path": "/home/user/private/usage.ipynb"},
    })

    assert "See the usage notebook" in rendered
    assert "/home/" not in rendered


def test_quick_start_preserves_release_relative_notebook_path():
    rendered = stage3._render_quick_start({
        "paths": {"notebook_path": "examples/usage.ipynb"},
    })

    assert "[Usage Notebook](examples/usage.ipynb)" in rendered


def test_generate_uses_config_basename_in_release_footer(tmp_path: Path):
    dataset_root = tmp_path / "neurotoken"
    dataset_root.mkdir()
    (dataset_root / "manifest.json").write_text(
        '{"modality":"meg","num_sessions":0,"sessions":[]}',
        encoding="utf-8",
    )
    cfg = {
        "dataset_name": "Example",
        "short_description": "Example release.",
        "source_dataset": {"name": "Example", "modality": "meg"},
        "documentation": {"version": "1.0"},
        "scope": {},
        "paths": {"notebook_path": "/share/private/usage.ipynb"},
    }

    markdown, _warnings = stage3._generate(
        cfg,
        dataset_root,
        max_depth=1,
        exclude_dirs=set(),
        cfg_path="/home/user/run/configs/pipeline_config.toml",
    )

    assert "Configuration: `pipeline_config.toml`" in markdown
    assert "/home/" not in markdown
    assert "/share/" not in markdown
