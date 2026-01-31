from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from datasets import load_dataset

def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_text(x: Any) -> Optional[str]:
    if x is None:
        return None
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()
    return x if x else None


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def load_cfg(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config introuvable: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("La config doit être un mapping YAML.")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    hf = cfg["hf"]
    out_cfg = cfg["output"]
    pre = cfg.get("preprocess", {})
    fmt = cfg.get("formatting", {})

    dataset_name = hf["dataset_name"]
    subset = hf.get("subset")
    split = hf.get("split", "train")

    out_dir = ensure_dir(out_cfg.get("out_dir", "staging_data/admin_multi_term"))
    prefix = out_cfg.get("prefix", "admin")

    # Load dataset from HF
    ds = load_dataset(dataset_name, subset, split=split)

    # Optional shuffle / subsample
    if pre.get("shuffle", False):
        ds = ds.shuffle(seed=int(pre.get("seed", 42)))

    max_rows = pre.get("max_rows")
    if max_rows is not None:
        max_rows = int(max_rows)
        ds = ds.select(range(min(max_rows, len(ds))))

    min_prompt_len = int(pre.get("min_prompt_len", 1))
    min_term_len = int(pre.get("min_term_len", 1))
    min_def_len = int(pre.get("min_def_len", 1))

    sep = fmt.get("prompt_term_separator", "\n\nTerme: ")

    normalized_path = out_dir / f"{prefix}_normalized.jsonl"
    opt1_path = out_dir / f"{prefix}_pairs_prompt_to_def.jsonl"
    opt2_path = out_dir / f"{prefix}_pairs_term_to_def.jsonl"
    opt3_path = out_dir / f"{prefix}_pairs_prompt_term_to_def.jsonl"
    meta_path = out_dir / f"{prefix}_meta.json"

    # ---- Build normalized rows
    normalized_rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        prompt = clean_text(ex.get("prompt"))
        terms = ex.get("term")
        defs = ex.get("definition")

        if prompt is None or len(prompt) < min_prompt_len:
            continue
        if not isinstance(terms, list) or not isinstance(defs, list):
            continue

        # align by index; keep only pairs where both exist
        m = min(len(terms), len(defs))
        for k in range(m):
            t = clean_text(terms[k])
            d = clean_text(defs[k])
            if t is None or d is None:
                continue
            if len(t) < min_term_len or len(d) < min_def_len:
                continue

            normalized_rows.append({
                "source_row_id": i,
                "k": k,
                "prompt": prompt,
                "term": t,
                "definition": d,
            })

    n_norm = write_jsonl(normalized_path, normalized_rows)

    # ---- Build pairs (3 options) from normalized
    # dedupe = bool(pre.get("dedupe_pairs", True))

    # def make_pairs_opt1(rows: List[Dict[str, Any]]):
    #     seen = set()
    #     for r in rows:
    #         q = r["prompt"]
    #         p = r["definition"]
    #         key = (q, p)
    #         if dedupe:
    #             h = sha1_str(q + "\n" + p)
    #             if h in seen:
    #                 continue
    #             seen.add(h)
    #         yield {"query": q, "positive": p}

    # def make_pairs_opt2(rows: List[Dict[str, Any]]):
    #     seen = set()
    #     for r in rows:
    #         q = r["term"]
    #         p = r["definition"]
    #         if dedupe:
    #             h = sha1_str(q + "\n" + p)
    #             if h in seen:
    #                 continue
    #             seen.add(h)
    #         yield {"query": q, "positive": p}

    # def make_pairs_opt3(rows: List[Dict[str, Any]]):
    #     seen = set()
    #     for r in rows:
    #         q = r["prompt"] + sep + r["term"]
    #         p = r["definition"]
    #         if dedupe:
    #             h = sha1_str(q + "\n" + p)
    #             if h in seen:
    #                 continue
    #             seen.add(h)
    #         yield {"query": q, "positive": p}

    # n_opt1 = write_jsonl(opt1_path, make_pairs_opt1(normalized_rows))
    # n_opt2 = write_jsonl(opt2_path, make_pairs_opt2(normalized_rows))
    # n_opt3 = write_jsonl(opt3_path, make_pairs_opt3(normalized_rows))

    # meta = {
    #     "dataset": {"name": dataset_name, "subset": subset, "split": split, "rows_loaded": len(ds)},
    #     "preprocess": pre,
    #     "formatting": {"prompt_term_separator": sep},
    #     "outputs": {
    #         "normalized": {"path": str(normalized_path), "rows": n_norm},
    #         "pairs_prompt_to_def": {"path": str(opt1_path), "rows": n_opt1},
    #         "pairs_term_to_def": {"path": str(opt2_path), "rows": n_opt2},
    #         "pairs_prompt_term_to_def": {"path": str(opt3_path), "rows": n_opt3},
    #     },
    #     "notes": [
    #         "chosen/rejected ignorés pour les embeddings.",
    #         "Une ligne normalisée = un (term, definition) aligné par index dans term/definition.",
    #     ],
    # }
    # meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # print("OK - fichiers générés :")
    # print(f"- {normalized_path} ({n_norm})")
    # print(f"- {opt1_path} ({n_opt1})")
    # print(f"- {opt2_path} ({n_opt2})")
    # print(f"- {opt3_path} ({n_opt3})")
    # print(f"- {meta_path}")

    # ---- Build pairs (3 options) from normalized
    dedupe = bool(pre.get("dedupe_pairs", True))

    opt1_rich_path = out_dir / f"{prefix}_pairs_prompt_to_def.rich.jsonl"
    opt2_rich_path = out_dir / f"{prefix}_pairs_term_to_def.rich.jsonl"
    opt3_rich_path = out_dir / f"{prefix}_pairs_prompt_term_to_def.rich.jsonl"

    opt1_train_path = out_dir / f"{prefix}_pairs_prompt_to_def.train.jsonl"
    opt2_train_path = out_dir / f"{prefix}_pairs_term_to_def.train.jsonl"
    opt3_train_path = out_dir / f"{prefix}_pairs_prompt_term_to_def.train.jsonl"

    def make_opt1_rich(rows):
        seen = set()
        for r in rows:
            q = r["prompt"]
            pos = r["definition"]
            t = r["term"]
            key = q + "\n" + pos
            if dedupe:
                h = sha1_str(key)
                if h in seen:
                    continue
                seen.add(h)
            yield {
                "pair_mode": "prompt_to_definition",
                "query": q,
                "positive": pos,
                "term": t,
                "prompt": r["prompt"],
                "definition": r["definition"],
                "source_row_id": r["source_row_id"],
                "k": r["k"],
            }

    def make_opt2_rich(rows):
        seen = set()
        for r in rows:
            q = r["term"]
            pos = r["definition"]
            t = r["term"]
            key = q + "\n" + pos
            if dedupe:
                h = sha1_str(key)
                if h in seen:
                    continue
                seen.add(h)
            yield {
                "pair_mode": "term_to_definition",
                "query": q,
                "positive": pos,
                "term": t,
                "prompt": r["prompt"],
                "definition": r["definition"],
                "source_row_id": r["source_row_id"],
                "k": r["k"],
            }

    def make_opt3_rich(rows):
        seen = set()
        for r in rows:
            q = r["prompt"] + sep + r["term"]
            pos = r["definition"]
            t = r["term"]
            key = q + "\n" + pos
            if dedupe:
                h = sha1_str(key)
                if h in seen:
                    continue
                seen.add(h)
            yield {
                "pair_mode": "prompt_term_to_definition",
                "query": q,
                "positive": pos,
                "term": t,
                "prompt": r["prompt"],
                "definition": r["definition"],
                "source_row_id": r["source_row_id"],
                "k": r["k"],
            }

    def to_train(rows):
        for r in rows:
            yield {"query": r["query"], "positive": r["positive"]}

    n_opt1_rich = write_jsonl(opt1_rich_path, make_opt1_rich(normalized_rows))
    n_opt2_rich = write_jsonl(opt2_rich_path, make_opt2_rich(normalized_rows))
    n_opt3_rich = write_jsonl(opt3_rich_path, make_opt3_rich(normalized_rows))

    n_opt1_train = write_jsonl(opt1_train_path, to_train(_read_jsonl(opt1_rich_path)))
    n_opt2_train = write_jsonl(opt2_train_path, to_train(_read_jsonl(opt2_rich_path)))
    n_opt3_train = write_jsonl(opt3_train_path, to_train(_read_jsonl(opt3_rich_path)))

    print("OK - fichiers générés :")
if __name__ == "__main__":
    main()
