from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


REQUIRED_PATHS = [
    ("exp_id",),
    ("model", "base_model_ref"),
    ("task", "type"),
    ("task", "format"),
    ("task", "text_fields", "query"),
    ("task", "text_fields", "positive"),
    ("data", "incoming_dir"),
]


def _get(d: Dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(".".join(path))
        cur = cur[k]
    return cur


def load_exp_config(config_path: str | Path) -> Dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config introuvable: {p}")

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("Le YAML doit contenir un objet (mapping) au root.")

    # Validate required fields
    missing = []
    for path in REQUIRED_PATHS:
        try:
            _get(cfg, path)
        except KeyError:
            missing.append(".".join(path))

    if missing:
        raise ValueError(f"Champs manquants dans la config: {missing}")

    # Basic normalization / defaults
    cfg.setdefault("tracking", {})
    cfg["tracking"].setdefault("local", {"enabled": True})
    cfg["tracking"].setdefault("wandb", {"enabled": False})

    cfg.setdefault("runtime", {})
    cfg["runtime"].setdefault("resume_if_interrupted", True)

    # Ensure task constraints (for now)
    if cfg["task"]["type"] != "embedding":
        raise ValueError("Seul task.type=embedding est supporté dans ce MVP.")
    if cfg["task"]["format"] not in {"pair", "triplet"}:
        raise ValueError("task.format doit être 'pair' ou 'triplet'.")

    return cfg
