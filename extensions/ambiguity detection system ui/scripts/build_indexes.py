import json
import yaml
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_datasets(cfg: dict) -> List[Dict]:
    ds_dir = Path(cfg["datasets_dir"])
    mode = cfg.get("datasets_mode", "scan")

    if mode == "scan":
        return [{"id": p.stem, "path": p} for p in sorted(ds_dir.glob("*.jsonl"))]

    return cfg["datasets"]


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            obj["_row_index"] = i
            rows.append(obj)
    return rows


def build_text(row: dict, cfg: dict) -> str:
    mode = cfg.get("text_mode", "positive")

    if mode == "positive":
        return row.get("positive", "").strip()

    fields = cfg.get("text_fields", {})
    parts = []
    for k in fields.values():
        if k in row and row[k]:
            parts.append(str(row[k]).strip())
    return " ".join(parts)


def index_exists(out_dir: Path) -> bool:
    return (
        (out_dir / "faiss.index").exists()
        and (out_dir / "meta.jsonl").exists()
        and (out_dir / "embeddings.npy").exists()
    )


def build_faiss_index(vectors: np.ndarray, normalize: bool):
    if normalize:
        faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def index_one_model_dataset(
    model_path: Path,
    scope: str,
    model_id: str,
    dataset_id: str,
    dataset_path: Path,
    cfg: dict,
):
    out_dir = Path(cfg["index_root"]) / scope / model_id / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if index_exists(out_dir):
        print(f"[SKIP] {scope}/{model_id}/{dataset_id}")
        return

    print(f"[INDEX] {scope}/{model_id}/{dataset_id}")

    rows = load_jsonl(dataset_path)

    texts = []
    metas = []

    for r in rows:
        text = build_text(r, cfg)
        if not text:
            continue
        texts.append(text)
        metas.append({
            "row_index": r.get(cfg.get("id_field", "_row_index"), r["_row_index"]),
            "raw": r,
        })

    if not texts:
        print(f"[WARN] empty dataset after filtering: {dataset_id}")
        return

    model = SentenceTransformer(
        str(model_path),
        device=cfg.get("device", "cpu"),
    )

    # ✅ CORRECT: définir la longueur max ici
    model.max_seq_length = cfg.get("max_length", 256)

    embeddings = model.encode(
        texts,
        batch_size=cfg.get("batch_size", 32),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=cfg.get("faiss", {}).get("normalize", True),
    ).astype("float32")

    np.save(out_dir / "embeddings.npy", embeddings)

    index = build_faiss_index(
        embeddings,
        normalize=cfg.get("faiss", {}).get("normalize", True),
    )
    faiss.write_index(index, str(out_dir / "faiss.index"))

    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    info = {
        "scope": scope,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "num_items": len(metas),
        "embedding_dim": embeddings.shape[1],
        "max_length": model.max_seq_length,
        "text_mode": cfg.get("text_mode"),
    }

    with open(out_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def main():
    cfg = load_yaml("config/indexing.yaml")

    base_dir = Path(cfg["models"]["base_dir"])
    merged_dir = Path(cfg["models"]["merged_dir"])

    datasets = list_datasets(cfg)

    for scope, models_root in [("base", base_dir), ("merged", merged_dir)]:
        if not models_root.exists():
            continue

        for model_path in models_root.iterdir():
            if not model_path.is_dir():
                continue

            model_id = model_path.name

            for ds in datasets:
                index_one_model_dataset(
                    model_path=model_path,
                    scope=scope,
                    model_id=model_id,
                    dataset_id=ds["id"],
                    dataset_path=ds["path"],
                    cfg=cfg,
                )


if __name__ == "__main__":
    main()