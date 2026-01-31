from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.io import ensure_dir, write_json, write_yaml, append_jsonl


def _try_read_json(path: Path) -> Any | None:
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _snapshot_seq_id(snapshots_root: Path) -> int:
    if not snapshots_root.exists():
        return 1
    existing = [p for p in snapshots_root.iterdir() if p.is_dir() and p.name.startswith("ds_")]
    return len(existing) + 1


def make_snapshot_id(exp_id: str, split: Optional[str], snapshots_root: Path) -> str:
    """
    Exemple: ds_0001_initial, ds_0002_iter01, ds_0003_unknown
    """
    n = _snapshot_seq_id(snapshots_root)
    suffix = (split or "unknown").strip() or "unknown"
    return f"ds_{n:04d}_{suffix}"

def create_snapshot_from_incoming(
    exp_id: str,
    incoming_batch_path: Path,
    incoming_meta: Dict[str, Any],
    task_cfg: Dict[str, Any],
    snapshots_root: Path,
    registry_dir: Path,
    processed_root: Path,
    commit: bool = False,
    snapshot_id: str | None = None,
) -> Dict[str, Any]:
    ensure_dir(snapshots_root)
    ensure_dir(processed_root)
    ensure_dir(registry_dir)

    split = incoming_meta.get("split")
    proc_dir = None

    if commit:
        if not snapshot_id:
            raise ValueError("commit=True exige snapshot_id")
    else:
        if not snapshot_id:
            snapshot_id = make_snapshot_id(exp_id, split, snapshots_root)

    snapshot_dir = ensure_dir(snapshots_root / snapshot_id)
    data_dir = ensure_dir(snapshot_dir / "data")

    dst_train = data_dir / "train.jsonl"

    # fingerprint basé sur incoming (ok)
    fp = sha256_file(incoming_batch_path)

    if not commit:
        # stage: copie unique
        dst_train.write_bytes(incoming_batch_path.read_bytes())

        manifest = {
            "schema_version": "1.0",
            "snapshot_id": snapshot_id,
            "exp_id": exp_id,
            "created_at": _utc_now(),
            "intent": "initial" if split == "initial" else "incremental",
            "delta_of": None,
            "source": {
                "type": incoming_meta.get("provider") or "unknown",
                "incoming_file": str(incoming_batch_path).replace("\\", "/"),
                "meta": incoming_meta,
            },
            "task": task_cfg,
            "canonical": {
                "format": "jsonl",
                "data_files": ["data/train.jsonl"],
                "stats": {"num_samples": _count_lines(dst_train)},
            },
            "fingerprint": {"algo": "sha256", "value": fp},
            "state": "STAGED",
        }
        write_json(snapshot_dir / "manifest.json", manifest)
    else:
        # commit: update manifest state only
        manifest_path = snapshot_dir / "manifest.json"
        manifest = _try_read_json(manifest_path) or {}
        manifest["state"] = "COMMITTED"
        write_json(manifest_path, manifest)

        # registry datasets.jsonl
        record = {
            "schema_version": "1.0",
            "type": "dataset_snapshot",
            "created_at": manifest.get("created_at") or _utc_now(),
            "exp_id": exp_id,
            "snapshot_id": snapshot_id,
            "snapshot_dir": str(snapshot_dir).replace("\\", "/"),
            "fingerprint": manifest.get("fingerprint") or {"algo": "sha256", "value": fp},
            "num_samples": (manifest.get("canonical", {}).get("stats", {}) or {}).get("num_samples"),
            "split": split,
            "provider": incoming_meta.get("provider"),
            "simulation_id": incoming_meta.get("simulation_id"),
        }
        append_jsonl(registry_dir / "datasets.jsonl", record)

        # déplacer incoming -> processed/<snapshot_id>/
        proc_dir = ensure_dir(processed_root / snapshot_id)
        proc_batch = proc_dir / incoming_batch_path.name
        proc_batch.write_bytes(incoming_batch_path.read_bytes())
        incoming_batch_path.unlink(missing_ok=True)

        meta_file = incoming_batch_path.parent / (incoming_batch_path.name + ".meta.json")
        if meta_file.exists():
            (proc_dir / meta_file.name).write_bytes(meta_file.read_bytes())
            meta_file.unlink(missing_ok=True)

    return {
        "snapshot_id": snapshot_id,
        "snapshot_dir": str(snapshot_dir).replace("\\", "/"),
        "manifest": manifest,
        "fingerprint": fp,
        "processed_dir": str(proc_dir).replace("\\", "/") if proc_dir else None,
    }

def _count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def create_snapshot_from_incoming_old(
    exp_id: str,
    incoming_batch_path: Path,
    incoming_meta: Dict[str, Any],
    task_cfg: Dict[str, Any],
    snapshots_root: Path,
    registry_dir: Path,
    processed_root: Path,
    commit: bool=False,
    snapshot_id: str | None = None
) -> Dict[str, Any]:
    """
    - crée un dossier snapshot immuable
    - écrit manifest.json
    - copie data/train.jsonl dans snapshot
    - déplace original vers processed/<snapshot_id>/
    - enregistre dans registry/datasets.jsonl
    Retourne un dict avec snapshot_id, snapshot_dir, manifest, fingerprint.
    """
    ensure_dir(snapshots_root)
    ensure_dir(processed_root)

    split = incoming_meta.get("split")
    snapshot_id = make_snapshot_id(exp_id, split, snapshots_root)

    snapshot_dir = ensure_dir(snapshots_root / snapshot_id)
    data_dir = ensure_dir(snapshot_dir / "data")

    # fingerprint du batch
    fp = sha256_file(incoming_batch_path)

    # copier le batch comme train.jsonl
    dst_train = data_dir / "train.jsonl"
    dst_train.write_bytes(incoming_batch_path.read_bytes())

    # manifest
    manifest = {
        "schema_version": "1.0",
        "snapshot_id": snapshot_id,
        "exp_id": exp_id,
        "created_at": _utc_now(),
        "intent": "initial" if split == "initial" else "incremental",
        "delta_of": None,  # on pourra remplir plus tard (ex: dernier snapshot)
        "source": {
            "type": incoming_meta.get("provider") or "unknown",
            "incoming_file": str(incoming_batch_path).replace("\\", "/"),
            "meta": incoming_meta,
        },
        "task": task_cfg,
        "canonical": {
            "format": "jsonl",
            "data_files": ["data/train.jsonl"],
            "stats": {
                "num_samples": _count_lines(dst_train),
            },
        },
        "fingerprint": {
            "algo": "sha256",
            "value": fp,
        },
        "state": "COMMITTED" if commit else "STAGED",

    }
    write_json(snapshot_dir / "manifest.json", manifest)

    # registry datasets.jsonl (append-only)
    record = {
        "schema_version": "1.0",
        "type": "dataset_snapshot",
        "created_at": manifest["created_at"],
        "exp_id": exp_id,
        "snapshot_id": snapshot_id,
        "snapshot_dir": str(snapshot_dir).replace("\\", "/"),
        "fingerprint": manifest["fingerprint"],
        "num_samples": manifest["canonical"]["stats"]["num_samples"],
        "split": split,
        "provider": incoming_meta.get("provider"),
        "simulation_id": incoming_meta.get("simulation_id"),
    }
    if commit:
        if not snapshot_id:
            raise ValueError("commit=True exige snapshot_id")
        # append registry?datasets.jsonl
        append_jsonl(registry_dir / "datasets.jsonl", record)

        # déplacer incoming -> processed/<snapshot_id>/
        proc_dir = ensure_dir(processed_root / snapshot_id)
        proc_batch = proc_dir / incoming_batch_path.name
        proc_batch.write_bytes(incoming_batch_path.read_bytes())
        incoming_batch_path.unlink(missing_ok=True)

        # déplacer meta aussi si présent
        meta_path_str = incoming_meta.get("incoming_file_meta_path")  # pas utilisé
        # si meta file existe à côté, on la cherche par convention
        meta_file = incoming_batch_path.parent / (incoming_batch_path.name + ".meta.json")
        if meta_file.exists():
            (proc_dir / meta_file.name).write_bytes(meta_file.read_bytes())
            meta_file.unlink(missing_ok=True)

    else:
        if not snapshot_id:
            snapshot_id = make_snapshot_id(...)
        proc_dir

    return {
        "snapshot_id": snapshot_id,
        "snapshot_dir": str(snapshot_dir).replace("\\", "/"),
        "manifest": manifest,
        "fingerprint": fp,
        "processed_dir": str(proc_dir).replace("\\", "/"),
    }


