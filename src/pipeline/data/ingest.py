from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class IncomingBatch:
    batch_path: Path
    meta_path: Optional[Path]
    meta: Dict[str, Any]


def load_incoming_batch(incoming_dir: Path, batch_filename: str = "batch.jsonl") -> Optional[IncomingBatch]:
    """
    Retourne le batch + meta si présent.
    Si batch absent: None.
    """
    batch_path = incoming_dir / batch_filename
    if not batch_path.exists():
        return None

    meta_path = incoming_dir / f"{batch_filename}.meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    return IncomingBatch(batch_path=batch_path, meta_path=meta_path if meta_path.exists() else None, meta=meta)
