from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config import load_exp_config
from .utils.io import ensure_dir, write_json, write_yaml
from .registry.registry import init_registry, load_index, save_index, register_run

from .data.ingest import load_incoming_batch
from .data.snapshot import create_snapshot_from_incoming
from .train.embedding_lora import TrainConfig, train_embedding_lora_chain

from src.pipeline.model_store import ensure_base_model_local

import json
import shutil



def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _try_read_json(path: Path):
    try:
        import json
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_run_id(run_num: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"run_{run_num:04d}_{ts}"


def run_experiment(config_path: str | Path, dry_run: bool = True) -> Dict[str, Any]:
    cfg = load_exp_config(config_path)
    exp_id = cfg["exp_id"]

    produced_model_id = None

    # registry
    registry_dir = cfg.get("tracking", {}).get("local", {}).get("registry_dir", "registry")
    reg = init_registry(registry_dir)
    index = load_index(reg)

    # experiment dirs
    exp_root = ensure_dir(Path("experiments") / exp_id)
    runs_root = ensure_dir(exp_root / "runs")
    ensure_dir(exp_root / "datasets" / "snapshots")
    ensure_dir(exp_root / "models" / "lora_adapters")

    # choose run number from index or count existing
    existing_runs = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    run_num = len(existing_runs) + 1
    run_id = _make_run_id(run_num)

    run_dir = ensure_dir(runs_root / run_id)
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "artifacts")

    # incoming meta (si feeder)
    incoming_dir_tpl = cfg.get("data", {}).get("incoming_dir", f"data_sources/{exp_id}/incoming")
    incoming_dir = Path(incoming_dir_tpl.replace("{exp_id}", exp_id))
    batch_filename = cfg.get("feeder", {}).get("batch_filename", "batch.jsonl")
    # incoming_meta = _try_read_json(incoming_dir / f"{batch_filename}.meta.json") or {}

     # --- Incoming -> Snapshot
    incoming_dir_tpl = cfg.get("data", {}).get("incoming_dir", f"data_sources/{exp_id}/incoming")
    incoming_dir = Path(incoming_dir_tpl.replace("{exp_id}", exp_id))
    ensure_dir(incoming_dir)

    feeder_cfg = cfg.get("feeder", {})
    batch_filename = feeder_cfg.get("batch_filename", "batch.jsonl")

    batch = load_incoming_batch(incoming_dir, batch_filename=batch_filename)
    incoming_meta = batch.meta or {}

    snapshot_info = None
    if batch is not None:
        snapshots_root = exp_root / "datasets" / "snapshots"
        processed_root = incoming_dir.parent / "processed"  # data_sources/<exp_id>/processed

        snapshot_info = create_snapshot_from_incoming(
            exp_id=exp_id,
            incoming_batch_path=batch.batch_path,
            incoming_meta=batch.meta or {},
            task_cfg=cfg["task"],
            snapshots_root=snapshots_root,
            registry_dir=Path(reg),
            processed_root=processed_root,
            commit=False,
        )
    if batch is None:
        return {"exp_id": exp_id, "state": "SKIPPED", "reason": "incoming vide"}

    # write run_config.yaml (minimal)
    run_config = {
        "schema_version": "1.0",
        "run_id": run_id,
        "exp_id": exp_id,
        "created_at": _utc_now(),
        "dry_run": dry_run,
        "inputs": {
            "parent_model_ref": cfg["model"]["base_model_ref"],
            "train_snapshot_ids": [snapshot_info["snapshot_id"]] if snapshot_info else [],
            "eval_snapshot_ids": []
        },
        "task": cfg["task"],
        "training": cfg.get("training", {}),
        "lora": cfg.get("lora", {"enabled": True}),
        "tracking": cfg.get("tracking", {}),
        "artifacts": {
            "run_dir": str(run_dir),
            "artifacts_dir": str(run_dir / "artifacts")
        },
        "data_lineage": {
            "provider": incoming_meta.get("provider"),
            "simulation_id": incoming_meta.get("simulation_id"),
            "injected_split": incoming_meta.get("split"),
            "max_iterations": incoming_meta.get("max_iterations"),
            "ui_asset_ref": incoming_meta.get("ui_asset_ref"),
            "incoming_file": incoming_meta.get("incoming_file"),
            "source_train_file": incoming_meta.get("source_train_file"),
        },

    }
    write_yaml(run_dir / "run_config.yaml", run_config)

    # status.json STARTED
    status_path = run_dir / "status.json"
    status = {
        "schema_version": "1.0",
        "run_id": run_id,
        "exp_id": exp_id,
        "state": "STARTED",
        "started_at": _utc_now(),
        "ended_at": None,
        "error": None
    }
    write_json(status_path, status)

    # DRY-RUN: no training, directly mark DONE
    if dry_run:
        status["state"] = "DONE"
        status["ended_at"] = _utc_now()
        write_json(status_path, status)
    else:
        try:
            # 1) vérifier train.enabled
            train_cfg = cfg.get("training", {})
            
            if not train_cfg.get("enabled", False):
                # pas de training → on termine proprement
                status["state"] = "DONE"
                status["ended_at"] = _utc_now()
                write_json(status_path, status)
            else:
                resolved_base = ensure_base_model_local(cfg["model"])
                # 2) chemin du dataset snapshot (delta)
                snap_id = snapshot_info["snapshot_id"]
                snap_train = exp_root / "datasets" / "snapshots" / snap_id / "data" / "train.jsonl"

                # 3) déterminer parent adapter (LoRA chain) depuis index
                parent_adapter_dir = None
                last_model_by_exp = index.get("last_model_by_exp", {})
                parent_model_id = last_model_by_exp.get(exp_id)
                if parent_model_id:
                    # on suppose que tu stockeras l'adapter dir dans un manifest modèle,
                    # mais pour démarrer simple: on pointe vers models/lora_adapters/<model_id>/adapter
                    candidate = exp_root / "models" / "lora_adapters" / parent_model_id / "adapter"
                    if candidate.exists():
                        parent_adapter_dir = candidate

                # 4) entraîner
                adapter_out = run_dir / "artifacts" / "lora_adapter"
                tc = TrainConfig(
                    base_model_ref=resolved_base,
                    max_seq_length=int(train_cfg.get("max_seq_length", 256)),
                    batch_size=int(train_cfg.get("batch_size", 16)),
                    lr=float(train_cfg.get("lr", 2e-4)),
                    epochs=int(train_cfg.get("epochs", 1)),
                    device=str(train_cfg.get("device", "cuda")),
                    precision=str(train_cfg.get("precision", "bf16")),
                )
                
                metrics = train_embedding_lora_chain(
                    train_jsonl=snap_train,
                    output_adapter_dir=adapter_out,
                    # base_model_ref=cfg["model"]["base_model_ref"],
                    base_model_ref=resolved_base,
                    lora_cfg=cfg.get("lora", {}),
                    cfg=tc,
                    parent_adapter_dir=parent_adapter_dir,
                )
                print(f"[TRAIN] done metrics={metrics}")
                # 5) enregistrer modèle (model_id + manifest)
                model_id = f"lora_{run_id}"
                produced_model_id = model_id
                model_dir = ensure_dir(exp_root / "models" / "lora_adapters" / model_id)
                # on copie l'adapter du run dans un emplacement stable du modèle
                stable_adapter_dir = ensure_dir(model_dir / "adapter")
                # copie simple (pas de symlink)
                for p in adapter_out.rglob("*"):
                    if p.is_dir():
                        continue
                    rel = p.relative_to(adapter_out)
                    dst = stable_adapter_dir / rel
                    ensure_dir(dst.parent)
                    dst.write_bytes(p.read_bytes())

                model_manifest = {
                    "schema_version": "1.0",
                    "model_id": model_id,
                    "exp_id": exp_id,
                    "run_id": run_id,
                    "created_at": _utc_now(),
                    "base_model_ref": cfg["model"]["base_model_ref"],
                    "base_model_local_dir": resolved_base,
                    "strategy": cfg["model"].get("strategy", "chain"),
                    "parent_model_id": parent_model_id,
                    "adapter_dir": str(stable_adapter_dir),
                    "train_snapshot_ids": [snap_id],
                    "metrics": metrics,
                    "data_lineage": run_config.get("data_lineage", {}),
                }
                write_json(model_dir / "manifest.json", model_manifest)

                # 6) registry/models.jsonl + update index last_model_by_exp
                model_record = {
                    "schema_version": "1.0",
                    "type": "model",
                    "created_at": model_manifest["created_at"],
                    "exp_id": exp_id,
                    "model_id": model_id,
                    "run_id": run_id,
                    "model_dir": str(model_dir),
                    "adapter_dir": str(stable_adapter_dir),
                    "parent_model_id": parent_model_id,
                    "train_snapshot_ids": [snap_id],
                    "metrics": metrics,
                }
                append_jsonl(Path(reg) / "models.jsonl", model_record)

                index.setdefault("last_model_by_exp", {})
                index["last_model_by_exp"][exp_id] = model_id
                save_index(reg, index)

                # COMMIT: consommer incoming seulement après succès training
                create_snapshot_from_incoming(
                    exp_id=exp_id,
                    incoming_batch_path=batch.batch_path,
                    incoming_meta=batch.meta or {},
                    task_cfg=cfg["task"],
                    snapshots_root=snapshots_root,
                    registry_dir=Path(reg),
                    processed_root=processed_root,
                    commit=True,
                    snapshot_id=snapshot_info["snapshot_id"]
                )

                # 7) terminer status
                status["state"] = "DONE"
                status["ended_at"] = _utc_now()
                write_json(status_path, status)

                print(f"[RUN] DONE run_id={run_id}")


        except Exception as e:
            status["state"] = "FAILED"
            status["ended_at"] = _utc_now()
            status["error"] = str(e)
            write_json(status_path, status)
            # rollback snapshot STAGED si training échoue
            # rollback snapshot STAGED
            try:
                snap_id = (snapshot_info or {}).get("snapshot_id")
                if snap_id:
                    snap_dir = exp_root / "datasets" / "snapshots" / snap_id
                    if snap_dir.exists():
                        shutil.rmtree(snap_dir)
            except Exception:
                pass

            # rollback adapter partiel
            try:
                import shutil
                adapter_out = run_dir / "artifacts" / "lora_adapter"
                if adapter_out.exists():
                    shutil.rmtree(adapter_out)
            except Exception:
                pass

            raise


    # registry record
    run_record = {
        "schema_version": "1.0",
        "type": "run",
        "created_at": run_config["created_at"],
        "exp_id": exp_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "state": status["state"],
        "dry_run": dry_run,
        "data_lineage": run_config.get("data_lineage", {}),
        "model_id": produced_model_id,
        # "model_id": index.get("last_model_by_exp", {}).get(exp_id),
       

    }
    register_run(reg, run_record)

    # update index
    if status["state"] == "DONE":
        index.setdefault("last_successful_run_by_exp", {})
        index["last_successful_run_by_exp"][exp_id] = run_id
        save_index(reg, index)

    return {
        "exp_id": exp_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "registry_dir": str(reg),
        "state": status["state"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Chemin vers configs/exp_*.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Exécute sans entraînement")
    args = parser.parse_args()

    result = run_experiment(args.config, dry_run=args.dry_run)
    print(result)


if __name__ == "__main__":
    main()
