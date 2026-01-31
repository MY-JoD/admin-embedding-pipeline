# tools/indexing/build_indexes.py
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModel

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


# -----------------------------
# Helpers I/O
# -----------------------------
def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # ligne invalide: skip
                continue


def mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: [B, T, H], attn_mask: [B, T]
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


# -----------------------------
# Model loading (base + optional adapter)
# -----------------------------
def load_encoder(base_model_ref: str, device: str, precision: str):
    tok = AutoTokenizer.from_pretrained(base_model_ref, use_fast=True)
    model = AutoModel.from_pretrained(base_model_ref)
    model.eval()

    # fallback device AVANT to()
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA demandé mais torch.cuda.is_available()=False -> fallback CPU")
        device = "cpu"
        if precision in ("bf16", "fp16"):
            precision = "fp32"

    # dtype
    if precision == "bf16":
        model = model.to(dtype=torch.bfloat16)
    elif precision == "fp16":
        model = model.to(dtype=torch.float16)
    else:
        model = model.to(dtype=torch.float32)

    # move device (OBLIGATOIRE)
    model = model.to(device)

    return tok, model

def apply_lora_adapter(model, adapter_dir: Path):
    # PEFT optionnel: si tu l’utilises déjà pour le train, tu l’as dans l’env
    from peft import PeftModel  # type: ignore
    return PeftModel.from_pretrained(model, str(adapter_dir))


# -----------------------------
# Index building
# -----------------------------
@dataclass
class UISchema:
    query_field: str
    text_field: str
    label_field: str
    id_field: str
    keep_fields: List[str]


@dataclass
class BuildCfg:
    batch_size: int
    max_length: int
    normalize: bool
    metric: str  # "ip"
    device: str
    precision: str


def encode_texts(
    tok,
    model,
    texts: List[str],
    max_length: int,
    device: str,
) -> torch.Tensor:
    with torch.no_grad():
        batch = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        emb = mean_pool(out.last_hidden_state, batch["attention_mask"])
        return emb


def build_faiss_index(vectors: torch.Tensor, normalize: bool) -> Any:
    # vectors: [N, D] torch
    vec = vectors.float().cpu().numpy()
    if normalize:
        faiss.normalize_L2(vec)

    d = vec.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vec)
    return index


def save_faiss(index: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


# -----------------------------
# Discovery: models + ui subsets
# -----------------------------
def list_models(exp_root: Path) -> List[Tuple[str, str, Optional[Path]]]:
    """
    Returns list of (model_ref, base_model_ref, adapter_dir)
    model_ref="base" has adapter_dir=None
    """
    models: List[Tuple[str, str, Optional[Path]]] = []

    # base model ref depuis le manifest des adapters si possible, sinon tu le passes via CLI/YAML
    # ici: on va lire un adapter manifest pour trouver base_model_ref si présent; sinon on forcera en CLI.
    adapters_root = exp_root / "models" / "lora_adapters"
    if adapters_root.exists():
        for d in adapters_root.iterdir():
            if not d.is_dir():
                continue
            mpath = d / "manifest.json"
            if not mpath.exists():
                continue
            m = read_json(mpath)
            model_id = m.get("model_id") or d.name
            adapter_dir = Path(m["adapter_dir"]) if "adapter_dir" in m else (d / "adapter")
            base_ref = m.get("base_model_ref")
            models.append((model_id, base_ref, adapter_dir))

    return models


def list_ui_subsets(ui_dir: Path, file_suffix: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in ui_dir.iterdir():
        if p.is_file() and p.name.endswith(file_suffix):
            # ex: initial.rich.jsonl -> subset = "initial"
            subset = p.name.replace(file_suffix, "")
            out[subset] = p
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


# -----------------------------
# Main build
# -----------------------------
def should_skip(manifest_path: Path, ui_sha: str, schema: UISchema, build: BuildCfg) -> bool:
    if not manifest_path.exists():
        return False
    try:
        m = read_json(manifest_path)
    except Exception:
        return False
    if m.get("ui_sha256") != ui_sha:
        return False
    if m.get("schema") != schema.__dict__:
        return False
    if m.get("build") != build.__dict__:
        return False
    # index file existence
    idx_path = manifest_path.parent / "faiss.index"
    meta_path = manifest_path.parent / "meta.jsonl"
    return idx_path.exists() and meta_path.exists()


def build_one_index(
    model_ref: str,
    base_model_ref: str,
    adapter_dir: Optional[Path],
    subset: str,
    ui_file: Path,
    out_dir: Path,
    schema: UISchema,
    cfg: BuildCfg,
) -> None:
    if not HAS_FAISS:
        raise RuntimeError("faiss introuvable. Installe faiss-cpu ou on bascule sur un autre backend.")

    ui_sha = sha256_file(ui_file)
    manifest_path = out_dir / "manifest.json"
    if should_skip(manifest_path, ui_sha, schema, cfg):
        print(f"[SKIP] {model_ref} / {subset} (up-to-date)")
        return

    print(f"[BUILD] {model_ref} / {subset} ui={ui_file.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # charger modèle
    tok, model = load_encoder(base_model_ref, cfg.device, cfg.precision)
    if adapter_dir is not None:
        model = apply_lora_adapter(model, adapter_dir)

    # lire docs
    docs: List[Dict[str, Any]] = []
    for row in iter_jsonl(ui_file):
        text = row.get(schema.text_field)
        label = row.get(schema.label_field)
        if text is None or label is None:
            continue

        # doc_id stable
        if schema.id_field in row and row.get(schema.id_field) is not None:
            doc_id = f"{subset}:{row[schema.id_field]}"
        else:
            # fallback: hash de (label + text)
            doc_id = hashlib.sha1((str(label) + "\n" + str(text)).encode("utf-8")).hexdigest()

        meta = {k: row.get(k) for k in schema.keep_fields}
        meta["doc_id"] = doc_id
        meta["subset"] = subset
        docs.append(meta)

    if not docs:
        print(f"[EMPTY] {model_ref}/{subset} no valid rows")
        return

    # encoder par batch
    all_vecs: List[torch.Tensor] = []
    bs = cfg.batch_size
    for i in range(0, len(docs), bs):
        chunk = docs[i : i + bs]
        texts = [str(d[schema.text_field]) for d in chunk]
        emb = encode_texts(tok, model, texts, cfg.max_length, cfg.device)
        all_vecs.append(emb)

        if (i // bs) % 10 == 0:
            print(f"  encoded {min(i+bs, len(docs))}/{len(docs)}")

    vecs = torch.cat(all_vecs, dim=0)

    # build index
    index = build_faiss_index(vecs, normalize=cfg.normalize)
    save_faiss(index, out_dir / "faiss.index")

    # write meta.jsonl aligned to vector order
    meta_path = out_dir / "meta.jsonl"
    if meta_path.exists():
        meta_path.unlink()
    for d in docs:
        append_jsonl(meta_path, d)

    # manifest
    manifest = {
        "schema_version": "1.0",
        "model_ref": model_ref,
        "base_model_ref": base_model_ref,
        "adapter_dir": str(adapter_dir).replace("\\", "/") if adapter_dir else None,
        "subset": subset,
        "ui_file": str(ui_file).replace("\\", "/"),
        "ui_sha256": ui_sha,
        "num_docs": len(docs),
        "schema": schema.__dict__,
        "build": cfg.__dict__,
    }
    write_json(manifest_path, manifest)
    print(f"[DONE] {model_ref}/{subset} num_docs={len(docs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--config", required=False, help="path yaml optionnel (si tu veux lire indexing.* depuis yaml)")
    ap.add_argument("--sim-id", required=True)
    ap.add_argument("--base-model-ref", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--file-suffix", default=".rich.jsonl")
    ap.add_argument("--text-field", default="positive")
    ap.add_argument("--label-field", default="term")
    ap.add_argument("--id-field", default="source_row_id")
    args = ap.parse_args()

    exp_root = Path("experiments") / args.exp_id
    if not exp_root.exists():
        raise SystemExit(f"exp introuvable: {exp_root}")

    ui_dir = exp_root / "assets_ui" / args.sim_id / "ui"
    if not ui_dir.exists():
        raise SystemExit(f"ui_dir introuvable: {ui_dir}")

    subsets = list_ui_subsets(ui_dir, args.file_suffix)
    if not subsets:
        raise SystemExit(f"aucun fichier UI trouvé dans {ui_dir}")

    # models: base + adapters
    adapters = list_models(exp_root)
    models: List[Tuple[str, str, Optional[Path]]] = [("base", args.base_model_ref, None)]
    for (model_id, base_ref, adapter_dir) in adapters:
        # si le manifest n’a pas base_model_ref, fallback sur CLI
        models.append((model_id, base_ref or args.base_model_ref, adapter_dir))

    schema = UISchema(
        query_field="query",
        text_field=args.text_field,
        label_field=args.label_field,
        id_field=args.id_field,
        keep_fields=["term", "query", "positive", "definition", "prompt", "pair_mode", "k", "source_row_id"],
    )
    build_cfg = BuildCfg(
        batch_size=args.batch_size,
        max_length=args.max_length,
        normalize=True,
        metric="ip",
        device=args.device,
        precision=args.precision,
    )

    out_root = exp_root / "indexes" / args.sim_id

    for (model_ref, base_ref, adapter_dir) in models:
        for subset, ui_file in subsets.items():
            out_dir = out_root / model_ref / subset
            build_one_index(
                model_ref=model_ref,
                base_model_ref=base_ref,
                adapter_dir=adapter_dir,
                subset=subset,
                ui_file=ui_file,
                out_dir=out_dir,
                schema=schema,
                cfg=build_cfg,
            )


if __name__ == "__main__":
    main()