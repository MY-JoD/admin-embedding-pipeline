from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_cfg(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def to_train_rows(rich_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # garde uniquement query/positive
    out = []
    for r in rich_rows:
        out.append({"query": r["query"], "positive": r["positive"]})
    return out


def split_by_ratios(rows: List[Dict[str, Any]], ratios: List[float]) -> List[List[Dict[str, Any]]]:
    # ratios doivent sommer ~1.0
    n = len(rows)
    counts = [int(math.floor(r * n)) for r in ratios]
    # corrige le reste dû aux arrondis
    diff = n - sum(counts)
    i = 0
    while diff > 0:
        counts[i % len(counts)] += 1
        diff -= 1
        i += 1

    chunks = []
    idx = 0
    for c in counts:
        chunks.append(rows[idx: idx + c])
        idx += c
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/simulate_*.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    rich_path = Path(cfg["input"]["rich_path"])
    sim_id = cfg["simulation"]["sim_id"]
    out_root = Path(cfg["output"]["out_dir"]) / sim_id

    n_iters = int(cfg["simulation"].get("n_iters", 4))
    initial_ratio = float(cfg["simulation"].get("initial_ratio", 0.5))
    shuffle = bool(cfg["simulation"].get("shuffle", True))
    seed = int(cfg["simulation"].get("seed", 42))

    if n_iters < 1:
        raise ValueError("n_iters doit être >= 1")
    if not (0.0 < initial_ratio < 1.0):
        raise ValueError("initial_ratio doit être entre 0 et 1")

    rows = read_jsonl(rich_path)
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(rows)

    # ratios: [initial] + n_iters itérations
    remain = 1.0 - initial_ratio
    per_iter = remain / n_iters
    ratios = [initial_ratio] + [per_iter] * n_iters

    parts = split_by_ratios(rows, ratios)
    initial_rows = parts[0]
    iter_rows_list = parts[1:]  # len == n_iters

    ui_dir = ensure_dir(out_root / "ui")
    train_dir = ensure_dir(out_root / "train")

    # write initial
    write_jsonl(ui_dir / "initial.rich.jsonl", initial_rows)
    write_jsonl(train_dir / "initial.train.jsonl", to_train_rows(initial_rows))

    # write iters
    for i, chunk in enumerate(iter_rows_list, start=1):
        write_jsonl(ui_dir / f"iter{i:02d}.rich.jsonl", chunk)
        write_jsonl(train_dir / f"iter{i:02d}.train.jsonl", to_train_rows(chunk))

    meta = {
        "sim_id": sim_id,
        "input_rich_path": str(rich_path),
        "num_rows_total": len(rows),
        "n_iters": n_iters,
        "initial_ratio": initial_ratio,
        "shuffle": shuffle,
        "seed": seed,
        "outputs": {
            "ui_dir": str(ui_dir),
            "train_dir": str(train_dir),
        },
        "counts": {
            "initial": len(initial_rows),
            **{f"iter{i:02d}": len(iter_rows_list[i-1]) for i in range(1, n_iters + 1)}
        }
    }
    ensure_dir(out_root)
    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK - simulation générée :")
    print(out_root)
    print(json.dumps(meta["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
