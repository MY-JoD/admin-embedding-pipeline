from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.io import ensure_dir, write_json, append_jsonl


def init_registry(registry_dir: str | Path) -> Path:
    reg = ensure_dir(registry_dir)
    ensure_dir(reg / "locks")

    # create files if not exist
    for name in ["datasets.jsonl", "runs.jsonl", "models.jsonl"]:
        p = reg / name
        if not p.exists():
            p.write_text("", encoding="utf-8")

    index_path = reg / "global_index.json"
    if not index_path.exists():
        write_json(index_path, {
            "schema_version": "1.0",
            "last_successful_run_by_exp": {},
            "last_model_by_exp": {},
            "last_snapshot_by_exp": {}
        })

    return reg


def load_index(registry_dir: str | Path) -> Dict[str, Any]:
    p = Path(registry_dir) / "global_index.json"
    if not p.exists():
        return {
            "schema_version": "1.0",
            "last_successful_run_by_exp": {},
            "last_model_by_exp": {},
            "last_snapshot_by_exp": {}
        }
    import json
    return json.loads(p.read_text(encoding="utf-8"))


def save_index(registry_dir: str | Path, index: Dict[str, Any]) -> None:
    write_json(Path(registry_dir) / "global_index.json", index)


def register_run(registry_dir: str | Path, record: Dict[str, Any]) -> None:
    append_jsonl(Path(registry_dir) / "runs.jsonl", record)
