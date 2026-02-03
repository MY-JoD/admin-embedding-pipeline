from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml  # pyyaml
except Exception as e:
    raise SystemExit("PyYAML manquant. Installe: pip install pyyaml") from e


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def load_rows_any(path: Path) -> List[Dict[str, Any]]:
    """
    Support:
      - .jsonl : json par ligne
      - .json  : liste d'objets OU objet {"data":[...]}
    """
    if path.suffix.lower() == ".jsonl":
        return list(iter_jsonl(path))

    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
        raise ValueError("JSON invalide: attendu liste ou {'data':[...]}")

    raise ValueError("Format non supporté. Utilise .jsonl ou .json")


def zfill_iter(n: int) -> str:
    return f"iter{n:02d}"


def detect_next_split(ui_dir: Path) -> str:
    ui_dir.mkdir(parents=True, exist_ok=True)
    initial = ui_dir / "initial.rich.jsonl"
    if not initial.exists():
        return "initial"

    # chercher iterXX existants
    max_i = 0
    for p in ui_dir.glob("iter*.rich.jsonl"):
        name = p.stem  # ex "iter02.rich" -> stem = "iter02.rich" (attention)
    # stem sur "iter02.rich.jsonl" donne "iter02.rich", donc on parse autrement:
    for p in ui_dir.iterdir():
        if not p.is_file():
            continue
        if not p.name.endswith(".rich.jsonl"):
            continue
        base = p.name.replace(".rich.jsonl", "")  # "iter02"
        if base.startswith("iter") and len(base) == 6 and base[3:].isdigit():
            max_i = max(max_i, int(base[3:]))

    return zfill_iter(max_i + 1)


def ensure_empty_or_force(dir_path: Path, force: bool) -> None:
    """
    incoming: si batch.jsonl existe, on refuse sauf force.
    """
    batch = dir_path / "batch.jsonl"
    meta = dir_path / "batch.jsonl.meta.json"
    if batch.exists() or meta.exists():
        if not force:
            raise SystemExit(
                f"incoming non vide: {batch} existe. Lance avec --force si tu veux écraser."
            )
        # wipe
        if batch.exists():
            batch.unlink()
        if meta.exists():
            meta.unlink()


@dataclass
class Mapping:
    query_field: str
    positive_field: str
    id_field: str
    label_field: str
    keep_fields: List[str]


def pick(row: Dict[str, Any], key: str) -> Any:
    return row.get(key)

def build_outputs(rows: List[Dict[str, Any]], mapping: Mapping):
    train_rows = []
    ui_rows = []

    for r in rows:
        q = r.get(mapping.query_field)
        p = r.get(mapping.positive_field)
        if q is None or p is None:
            continue

        train_rows.append({"query": q, "positive": p})

        ui = dict(r)  # garde tout
        ui["query"] = q
        ui["positive"] = p
        ui_rows.append(ui)

    return train_rows, ui_rows

def build_outputs2(
    rows: List[Dict[str, Any]],
    mapping: Mapping,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      - train_rows: [{"query":..., "positive":...}]
      - ui_rows: rich rows (keep_fields + id/label/query/positive)
    """
    train_rows: List[Dict[str, Any]] = []
    ui_rows: List[Dict[str, Any]] = []

    for r in rows:
        q = pick(r, mapping.query_field)
        p = pick(r, mapping.positive_field)
        if q is None or p is None:
            continue

        train_rows.append({"query": q, "positive": p})

        ui = {}
        # keep_fields
        for k in mapping.keep_fields:
            if k in r:
                ui[k] = r.get(k)

        # champs minimaux utiles UI
        ui["query"] = q
        ui["positive"] = p
        if mapping.label_field in r:
            ui[mapping.label_field] = r.get(mapping.label_field)
        if mapping.id_field in r:
            ui[mapping.id_field] = r.get(mapping.id_field)

        ui_rows.append(ui)

    return train_rows, ui_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/<exp>.yaml")
    ap.add_argument("--asset-id", required=True, help="ex: ui_0001_baconi_term_to_def")
    ap.add_argument("--input", required=True, help="path dataset .jsonl ou .json")
    ap.add_argument("--split", default=None, help="initial | iter01 | iter02 ... (optionnel)")
    ap.add_argument("--force", action="store_true", help="écrase incoming et/ou subset UI si existants")
    ap.add_argument("--status", action="store_true", help="affiche l'état et sort")
    ap.add_argument("--query-field", default=None)
    ap.add_argument("--positive-field", default=None)
    ap.add_argument("--id-field", default=None)
    ap.add_argument("--label-field", default=None)
    ap.add_argument("--keep-fields", default=None, help="liste séparée par virgule, optionnel")

    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = read_yaml(cfg_path)
    exp_id = cfg["exp_id"]

    # chemins
    exp_root = Path("experiments") / exp_id
    incoming_dir_tpl = cfg.get("data", {}).get("incoming_dir", f"data_sources/{exp_id}/incoming")
    incoming_dir = Path(str(incoming_dir_tpl).replace("{exp_id}", exp_id))
    ui_dir = exp_root / "assets_ui" / args.asset_id / "ui"
    meta_json_path = exp_root / "assets_ui" / args.asset_id / "meta.json"

    # mapping depuis YAML si possible (sinon via CLI)
    # On suppose un bloc optionnel dans yaml:
    # ui_schema: { query_field, positive_field, id_field, label_field, keep_fields }
    ui_schema = cfg.get("ui_schema", {}) or {}
    query_field = args.query_field or ui_schema.get("query_field") or "query"
    positive_field = args.positive_field or ui_schema.get("positive_field") or "positive"
    id_field = args.id_field or ui_schema.get("id_field") or "source_row_id"
    label_field = args.label_field or ui_schema.get("label_field") or "term"

    keep_fields = ui_schema.get("keep_fields")
    if args.keep_fields:
        keep_fields = [x.strip() for x in args.keep_fields.split(",") if x.strip()]
    if not keep_fields:
        keep_fields = ["term", "definition", "prompt", "pair_mode", "k", "source_row_id"]

    mapping = Mapping(
        query_field=query_field,
        positive_field=positive_field,
        id_field=id_field,
        label_field=label_field,
        keep_fields=keep_fields,
    )

    # STATUS
    if args.status:
        incoming_batch = incoming_dir / "batch.jsonl"
        print(f"exp_id: {exp_id}")
        print(f"incoming_dir: {incoming_dir}")
        print(f"incoming_present: {incoming_batch.exists()}")
        if incoming_batch.exists():
            print(f"  - {incoming_batch}")
        print(f"ui_dir: {ui_dir}")
        ui_dir.mkdir(parents=True, exist_ok=True)
        subsets = sorted([p.name for p in ui_dir.glob("*.rich.jsonl")])
        print(f"ui_subsets ({len(subsets)}):")
        for s in subsets:
            print(f"  - {s}")
        return

    # split
    if args.split:
        split = args.split.strip()
    else:
        split = detect_next_split(ui_dir)

    # input
    input_path = Path(args.input)
    rows = load_rows_any(input_path)
   
    # validation rapide: on exige query/positive dans le dataset
    missing = []
    for field in [query_field, positive_field]:
        if not any(field in r for r in rows):
            missing.append(field)
    if missing:
        raise SystemExit(f"dataset invalide: champs manquants: {missing}. Attendu au minimum query/positive.")
    
    train_rows, ui_rows = build_outputs(rows, mapping)

    if not train_rows:
        raise SystemExit("aucune ligne exploitable: vérifie query_field/positive_field")

    # 1) incoming
    incoming_dir.mkdir(parents=True, exist_ok=True)
    ensure_empty_or_force(incoming_dir, force=args.force)

    batch_path = incoming_dir / "batch.jsonl"
    batch_meta_path = incoming_dir / "batch.jsonl.meta.json"

    write_jsonl(batch_path, train_rows)

    meta = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "provider": "ui_upload",
        "exp_id": exp_id,
        "asset_id": args.asset_id,
        "split": split,
        "incoming_file": str(batch_path).replace("\\", "/"),
        "source_input": str(input_path).replace("\\", "/"),
        "counts": {
            "raw_rows": len(rows),
            "train_rows": len(train_rows),
            "ui_rows": len(ui_rows),
        },
        "mapping": {
            "query_field": mapping.query_field,
            "positive_field": mapping.positive_field,
            "id_field": mapping.id_field,
            "label_field": mapping.label_field,
            "keep_fields": mapping.keep_fields,
        },
    }
    write_json(batch_meta_path, meta)

    # 2) ui rich
    ui_dir.mkdir(parents=True, exist_ok=True)
    ui_file = ui_dir / f"{split}.rich.jsonl"
    if ui_file.exists() and not args.force:
        raise SystemExit(f"subset UI existe déjà: {ui_file}. Utilise --force pour écraser.")

    write_jsonl(ui_file, ui_rows)

    # 3) meta.json asset
    asset_meta = {}
    if meta_json_path.exists():
        try:
            asset_meta = json.loads(meta_json_path.read_text(encoding="utf-8"))
        except Exception:
            asset_meta = {}
    asset_meta.setdefault("schema_version", "1.0")
    asset_meta["asset_id"] = args.asset_id
    asset_meta["exp_id"] = exp_id
    asset_meta["updated_at"] = utc_now()
    asset_meta.setdefault("subsets", {})
    asset_meta["subsets"][split] = {
        "ui_file": str(ui_file).replace("\\", "/"),
        "num_ui_rows": len(ui_rows),
        "created_at": meta["created_at"],
    }
    asset_meta["mapping"] = meta["mapping"]
    write_json(meta_json_path, asset_meta)

    print("[OK] upload terminé")
    print(f"  split: {split}")
    print(f"  incoming: {batch_path}")
    print(f"  ui: {ui_file}")
    print("Ensuite: lance l'orchestrator sur cette exp pour consommer incoming et entraîner.")


if __name__ == "__main__":
    main()