from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional

from transformers import AutoModel, AutoTokenizer

def resolve_local_base_dir(model_cfg: dict) -> Path:
    store = Path(model_cfg.get("local_store_dir", "model_staging/base_models"))
    local_id = model_cfg.get("local_id") or model_cfg["base_model_ref"].replace("/", "__")
    return store / local_id

def ensure_base_model_local(model_cfg: dict) -> str:
    """
    Retourne un ref utilisable par transformers:
    - chemin local si présent (ou après download)
    - sinon le base_model_ref HF si tu n'actives pas la persistance
    """
    prefer_local = bool(model_cfg.get("prefer_local", True))
    allow_download = bool(model_cfg.get("allow_download", True))
    base_ref = model_cfg["base_model_ref"]

    local_dir = resolve_local_base_dir(model_cfg)

    if prefer_local and local_dir.exists():
        return str(local_dir)

    if not allow_download:
        raise RuntimeError(f"Base model introuvable en local et allow_download=false: {local_dir}")

    # Télécharge via HF cache puis persiste dans local_dir
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(base_ref)
    tok_ref = model_cfg.get("tokenizer_ref") or base_ref
    tok = AutoTokenizer.from_pretrained(tok_ref, use_fast=True)

    model.save_pretrained(str(local_dir))
    tok.save_pretrained(str(local_dir))

    return str(local_dir)