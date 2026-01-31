from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    dst.write_bytes(src.read_bytes())


def copy_tree(src_dir: Path, dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    for p in src_dir.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(src_dir)
        target = dst_dir / rel
        ensure_dir(target.parent)
        target.write_bytes(p.read_bytes())


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config introuvable: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("La config YAML doit être un mapping au root.")
    return cfg


def format_path(template: str, exp_id: str) -> str:
    return template.replace("{exp_id}", exp_id)


def split_name_for_state(state: Dict[str, Any]) -> str:
    return state.get("next_split", "initial")


def next_split(current: str, max_iterations: int) -> Optional[str]:
    # Sequence: initial -> iter01 -> ... -> iterXX -> None (done)
    if current == "initial":
        return "iter01" if max_iterations >= 1 else None
    if current.startswith("iter"):
        i = int(current.replace("iter", ""))
        if i >= max_iterations:
            return None
        return f"iter{i+1:02d}"
    raise ValueError(f"Split inconnu: {current}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-config", required=True, help="configs/exp_*.yaml")
    ap.add_argument("--force-split", default=None, help="Override split (initial|iter01|...)")
    args = ap.parse_args()

    cfg = load_yaml(args.exp_config)
    exp_id = cfg["exp_id"]

    feeder = cfg.get("feeder", {})
    if not feeder.get("enabled", True):
        print("Feeder désactivé dans la config. Rien à faire.")
        return
    if feeder.get("provider") != "simulation":
        print(f"provider={feeder.get('provider')} -> feeder simulation ne fait rien.")
        return

    max_iters = int(feeder.get("max_iterations", 4))
    batch_filename = feeder.get("batch_filename", "batch.jsonl")

    sim = feeder.get("simulation", {})
    sim_id = sim["sim_id"]
    sim_root = Path(sim.get("root_dir", "staging_data/simulations"))
    sim_dir = sim_root / sim_id

    train_dir = sim_dir / "train"
    ui_dir = sim_dir / "ui"
    meta_sim_path = sim_dir / "meta.json"

    if not train_dir.exists():
        raise FileNotFoundError(f"Simulation train introuvable: {train_dir}")
    if not meta_sim_path.exists():
        raise FileNotFoundError(f"meta.json introuvable: {meta_sim_path}")

    # data_sources/<exp_id>/
    incoming_dir = Path(format_path(cfg["data"]["incoming_dir"], exp_id))
    exp_ds_root = incoming_dir.parent  # data_sources/<exp_id>
    feeder_dir = ensure_dir(exp_ds_root / "feeder")
    state_path = feeder_dir / "state.json"

    # UI assets (copie une fois)
    ui_asset_dir_tpl = cfg.get("ui", {}).get("asset_dir", "experiments/{exp_id}/assets_ui")
    ui_asset_root = Path(format_path(ui_asset_dir_tpl, exp_id))
    ui_asset_ref = ui_asset_root / sim_id  # experiments/<exp_id>/assets_ui/<sim_id>

    # load/init state
    if state_path.exists():
        state = read_json(state_path)
    else:
        state = {
            "schema_version": "1.0",
            "exp_id": exp_id,
            "provider": "simulation",
            "simulation_id": sim_id,
            "max_iterations": max_iters,
            "next_split": "initial",
            "done": False,
            "history": [],
        }

    # sanity checks
    if state.get("done") is True:
        print("Simulation terminée (state.done=true). Rien à injecter.")
        return

    split = args.force_split or split_name_for_state(state)

    # enforce max_iterations
    if split.startswith("iter"):
        i = int(split.replace("iter", ""))
        if i > max_iters:
            state["done"] = True
            write_json(state_path, state)
            print(f"Split {split} > max_iterations={max_iters}. Terminé.")
            return

    src_train = train_dir / f"{split}.train.jsonl"
    if not src_train.exists():
        raise FileNotFoundError(f"Fichier train introuvable: {src_train}")

    # ensure dirs
    ensure_dir(incoming_dir)
    ensure_dir(exp_ds_root / "processed")

    # copy UI once (optional)
    if bool(sim.get("copy_ui_once", True)):
        ui_target = ui_asset_ref / "ui"
        if not ui_target.exists():
            if not ui_dir.exists():
                raise FileNotFoundError(f"Dossier UI simulation introuvable: {ui_dir}")
            copy_tree(ui_dir, ui_target)
            # copy meta
            copy_file(meta_sim_path, ui_asset_ref / "meta.json")

    # write incoming batch + meta
    dst_batch = incoming_dir / batch_filename

    # SAFE: ne pas injecter si incoming pas vide (sinon skip d'itérations)
    if dst_batch.exists():
        print("STOP: incoming contient déjà un batch. Lance d'abord l'orchestrator pour le consommer.")
        print(f"- batch existant: {dst_batch}")
        return
    copy_file(src_train, dst_batch)

    batch_meta = {
        "schema_version": "1.0",
        "exp_id": exp_id,
        "provider": "simulation",
        "simulation_id": sim_id,
        "split": split,
        "max_iterations": max_iters,
        "ui_asset_ref": str(ui_asset_ref).replace("\\", "/"),
        "created_at": utc_now(),
        "source_train_file": str(src_train).replace("\\", "/"),
        "incoming_file": str(dst_batch).replace("\\", "/"),
    }
    write_json(incoming_dir / f"{batch_filename}.meta.json", batch_meta)

    # update state
    state["history"].append({"split": split, "injected_at": utc_now(), "incoming": batch_filename})
    ns = next_split(split, max_iters)
    if ns is None:
        state["done"] = True
        state["next_split"] = None
    else:
        state["next_split"] = ns

    write_json(state_path, state)

    print("OK - injection effectuée")
    print(f"- split: {split}")
    print(f"- incoming: {dst_batch}")
    print(f"- meta: {dst_batch}.meta.json")
    print(f"- state: {state_path}")
    if batch_meta.get("ui_asset_ref"):
        print(f"- ui_asset_ref: {batch_meta['ui_asset_ref']}")


if __name__ == "__main__":
    main()
