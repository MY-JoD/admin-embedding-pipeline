from __future__ import annotations

import json
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import torch

import numpy as np

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

from transformers import AutoTokenizer, AutoModel


import numpy as np
import torch.nn.functional as F

try:
    from peft import PeftModel  # type: ignore
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False


st.set_page_config(page_title="Inference", layout="wide")
st.title("Inference")

ROOT = Path(".")
EXPS_DIR = ROOT / "experiments"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def list_exp_ids() -> List[str]:
    if not EXPS_DIR.exists():
        return []
    return sorted([p.name for p in EXPS_DIR.iterdir() if p.is_dir()])


def list_assets(exp_id: str) -> List[str]:
    p = EXPS_DIR / exp_id / "assets_ui"
    if not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir()])


def list_models_indexed(exp_id: str, asset_id: str) -> List[str]:
    p = EXPS_DIR / exp_id / "indexes" / asset_id
    if not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir()])


def read_json_safe(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return read_json(p)
    except Exception:
        return None


def build_model_display_map(exp_id: str) -> Dict[str, str]:
    """
    Retourne un mapping: model_ref -> label affiché
    - base => "V0 base (base_model_ref)"
    - loras triés par created_at => "V1 <model_id>", "V2 <model_id>", ...
    """
    exp_root = EXPS_DIR / exp_id
    out: Dict[str, str] = {}

    # base_model_ref: on va le chercher depuis un index manifest si possible
    base_ref = None
    # on tente de trouver n'importe quel manifest d'index existant
    idx_root = exp_root / "indexes"
    if idx_root.exists():
        for mpath in idx_root.rglob("manifest.json"):
            m = read_json_safe(mpath)
            if m and m.get("base_model_ref"):
                base_ref = m["base_model_ref"]
                break

    out["base"] = f"V0 base ({base_ref})" if base_ref else "V0 base"

    # loras: lire manifest modèles
    lora_root = exp_root / "models" / "lora_adapters"
    loras: List[Tuple[str, str]] = []  # (model_id, created_at)
    if lora_root.exists():
        for d in lora_root.iterdir():
            if not d.is_dir():
                continue
            mp = d / "manifest.json"
            m = read_json_safe(mp)
            if not m:
                continue
            mid = m.get("model_id") or d.name
            created = m.get("created_at") or ""
            loras.append((mid, created))

    # tri chronologique (string ISO)
    loras = sorted(loras, key=lambda x: (x[1], x[0]))
    for j, (mid, _) in enumerate(loras, start=1):
        out[mid] = f"V{j} {mid}"

    return out

def list_subsets(exp_id: str, asset_id: str, model_ref: str) -> List[str]:
    p = EXPS_DIR / exp_id / "indexes" / asset_id / model_ref
    if not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir()])


def mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom

def encode_query_norm(tok, model, text: str, max_length: int = 256) -> np.ndarray:
    with torch.no_grad():
        batch = tok([text], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        out = model(**batch)
        emb = mean_pool(out.last_hidden_state, batch["attention_mask"])[0].float()
        emb = F.normalize(emb, p=2, dim=0)
    return emb.cpu().numpy().astype("float32")

# @st.cache_resource(show_spinner=False)
# def load_encoder(base_model_ref: str):
#     tok = AutoTokenizer.from_pretrained(base_model_ref, use_fast=True)
#     model = AutoModel.from_pretrained(base_model_ref)
#     model.eval()
#     return tok, model


@st.cache_resource(show_spinner=False)
def load_encoder_any(base_model_ref: str, adapter_dir: Optional[str]):
    tok = AutoTokenizer.from_pretrained(base_model_ref, use_fast=True)
    model = AutoModel.from_pretrained(base_model_ref)
    model.eval()

    if adapter_dir:
        if not HAS_PEFT:
            raise RuntimeError("peft non installé mais adapter_dir présent.")
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()

    return tok, model


def encode_query(tok, model, text: str, max_length: int = 256) -> torch.Tensor:
    with torch.no_grad():
        batch = tok([text], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        out = model(**batch)
        emb = mean_pool(out.last_hidden_state, batch["attention_mask"])
        return emb[0].float().cpu()  # [D]


def faiss_search(index, qvec: torch.Tensor, k: int) -> Tuple[List[int], List[float]]:
    q = qvec.numpy().reshape(1, -1)
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    return idxs[0].tolist(), scores[0].tolist()


st.sidebar.subheader("Sélection")
exp_id = st.sidebar.selectbox("exp_id", options=list_exp_ids())

asset_opts = list_assets(exp_id) if exp_id else []
asset_id = st.sidebar.selectbox("asset_id", options=asset_opts) if asset_opts else None

model_opts = list_models_indexed(exp_id, asset_id) if (exp_id and asset_id) else []
model_display = build_model_display_map(exp_id) if exp_id else {}

model_ref = (
    st.sidebar.selectbox(
        "model_ref",
        options=model_opts,
        format_func=lambda x: model_display.get(x, x),
    )
    if model_opts else None
)

subset_opts = list_subsets(exp_id, asset_id, model_ref) if (exp_id and asset_id and model_ref) else []
subsets = st.sidebar.multiselect("subsets", options=subset_opts, default=subset_opts[:1]) if subset_opts else []

k = st.sidebar.slider("top-k", min_value=1, max_value=50, value=10, step=1)

st.sidebar.markdown("---")
max_len = st.sidebar.number_input("encode max_length", min_value=16, max_value=2048, value=256, step=16)


if not HAS_FAISS:
    st.error("faiss introuvable. Installe faiss-cpu.")
    st.stop()

if not (exp_id and asset_id and model_ref and subsets):
    st.info("Sélectionne exp_id / asset_id / model_ref / au moins un subset.")
    st.stop()

# idx_dir = EXPS_DIR / exp_id / "indexes" / asset_id / model_ref / subsets
# manifest_path = idx_dir / "manifest.json"
# index_path = idx_dir / "faiss.index"
# meta_path = idx_dir / "meta.jsonl"

idx_dirs = [EXPS_DIR / exp_id / "indexes" / asset_id / model_ref / s for s in subsets]


# vérifier que tous les subsets sélectionnés ont index+meta+manifest
bad = []
manifests = []
for d in idx_dirs:
    mp = d / "manifest.json"
    ip = d / "faiss.index"
    mep = d / "meta.jsonl"
    if not (mp.exists() and ip.exists() and mep.exists()):
        bad.append(str(d))
    else:
        manifests.append(read_json(mp))

if bad:
    st.error("Index incomplet pour: " + ", ".join(bad))
    st.stop()

# base_model_ref: prend celui du premier manifest
# base_model_ref = manifests[0]["base_model_ref"]
if bad:
    st.error("Index incomplet pour: " + ", ".join(bad))
    st.stop()

base_refs = {m.get("base_model_ref") for m in manifests}
if len(base_refs) != 1:
    st.error(f"base_model_ref différent entre subsets: {base_refs}")
    st.stop()
base_model_ref = next(iter(base_refs))

# base_refs = {m.get("base_model_ref") for m in manifests}
# if len(base_refs) != 1:
#     st.error(f"base_model_ref différent entre subsets: {base_refs}")
#     st.stop()

# base_model_ref = list(base_refs)[0]


base_refs = {m.get("base_model_ref") for m in manifests}
if len(base_refs) != 1:
    st.error(f"base_model_ref différent entre subsets: {base_refs}")
    st.stop()

st.subheader("Indexes sélectionnés")
c1, c2, c3 = st.columns(3)
c1.metric("model_ref", model_display.get(model_ref, model_ref))
c2.metric("subsets", ", ".join(subsets))
c3.metric("docs_total", int(sum(m.get("num_docs", 0) for m in manifests)))

with st.expander("manifests.json"):
    for s, m in zip(subsets, manifests):
        st.markdown(f"**{s}**")
        st.json(m)

query = st.text_area("Requête", height=120, placeholder="Tape ta requête…")
go = st.button("Rechercher", type="primary", use_container_width=True)

@st.cache_resource(show_spinner=False)
def load_meta_and_map(meta_path: str):
    rows = list(iter_jsonl(Path(meta_path)))
    m = {}
    for i, r in enumerate(rows):
        did = r.get("doc_id")
        if did:
            m[did] = i
    return rows, m

@st.cache_resource(show_spinner=False)
def load_index_obj(index_path: str):
    return faiss.read_index(index_path)

@st.cache_resource(show_spinner=False)
def load_one_index(path: Path):
    return faiss.read_index(str(path))

@st.cache_resource(show_spinner=False)
def load_one_meta(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))

@st.cache_resource(show_spinner=False)
def load_and_merge_indexes(idx_dirs_tuple: Tuple[str, ...]):
    """
    Charge plusieurs indexes FAISS (un par subset) et les combine via IndexShards.
    Avantages:
      - pas besoin d’accéder à .xb
      - supporte beaucoup plus de types d’index
      - IDs retournés sont globaux (offset par shard) si successive_ids=True
    """
    merged_meta: List[Dict[str, Any]] = []
    shard_indexes = []
    dim = None

    for d_str in idx_dirs_tuple:
        d = Path(d_str)
        index = faiss.read_index(str(d / "faiss.index"))
        meta = list(iter_jsonl(d / "meta.jsonl"))

        if dim is None:
            dim = index.d
        else:
            if index.d != dim:
                raise RuntimeError(f"Dimension mismatch: {index.d} vs {dim}")

        shard_indexes.append(index)
        merged_meta.extend(meta)

    # successive_ids=True => ids = concat (0..n1-1, n1..n1+n2-1, ...)
    shards = faiss.IndexShards(dim, True, True)
    for idx in shard_indexes:
        shards.add_shard(idx)

    return shards, merged_meta


index, meta = load_and_merge_indexes(tuple(str(d) for d in idx_dirs))

docid_to_global_idx = {}
for gi, row in enumerate(meta):
    did = row.get("doc_id")
    if did:
        docid_to_global_idx[did] = gi

compare_others = st.checkbox(
    "Afficher scores des autres modèles",
    key="show_others_scores",
    value=False,   # par défaut décoché
)


if go:
    # compare_others = st.checkbox("Afficher scores des autres modèles", value=True)

    if not query.strip():
        st.warning("Requête vide.")
        st.stop()

    with st.spinner("Encodage + recherche..."):
        # adapter_dir vient du manifest du modèle sélectionné
        adapter_dir = manifests[0].get("adapter_dir")

        tok, model = load_encoder_any(
            base_model_ref,
            adapter_dir
        )

        qvec = encode_query(tok, model, query, max_length=int(max_len))
        idxs, scores = faiss_search(index, qvec, int(k))


    # with st.spinner("Encodage + recherche..."):
    #     tok, model = load_encoder_any(base_model_ref)
    #     qvec = encode_query(tok, model, query, max_length=int(max_len))
    #     idxs, scores = faiss_search(index, qvec, int(k))
    
    all_models = list_models_indexed(exp_id, asset_id)  # ["base", "lora_run_x", ...]
    other_models = [m for m in all_models if m != model_ref]

    # prépare info index/manifest pour chaque modèle + subset
    model_subset_info = {}  # model -> list of dict {subset, index_path, meta_path, adapter_dir, base_model_ref}
    for mref in all_models:
        infos = []
        for s in subsets:
            d = EXPS_DIR / exp_id / "indexes" / asset_id / mref / s
            mp = d / "manifest.json"
            ip = d / "faiss.index"
            mep = d / "meta.jsonl"
            if not (mp.exists() and ip.exists() and mep.exists()):
                continue
            man = read_json(mp)
            infos.append({
                "subset": s,
                "manifest": man,
                "index_path": str(ip),
                "meta_path": str(mep),
                "base_model_ref": man.get("base_model_ref"),
                "adapter_dir": man.get("adapter_dir"),
            })
        model_subset_info[mref] = infos

    # sécurité: même base_model_ref pour les subsets d’un même modèle
    for mref, infos in model_subset_info.items():
        base_refs = {i["base_model_ref"] for i in infos if i.get("base_model_ref")}
        if len(base_refs) > 1:
            st.error(f"base_model_ref différent pour {mref}: {base_refs}")
            st.stop()

    def score_doc_for_model(mref: str, doc_id: str, query_text: str) -> Optional[float]:
        infos = model_subset_info.get(mref, [])
        if not infos:
            return None

        # encode query avec le bon modèle (base + adapter si présent)
        base_ref = infos[0]["base_model_ref"]
        # adapter_dir: pour base c'est None; pour lora c’est présent dans manifest
        adapter_dir = infos[0]["adapter_dir"]
        tok, mdl = load_encoder_any(base_ref, adapter_dir)
        qv = encode_query_norm(tok, mdl, query_text, max_length=int(max_len))  # (D,)

        # retrouver le doc_id dans le bon subset
        for info in infos:
            meta_rows, meta_map = load_meta_and_map(info["meta_path"])
            if doc_id not in meta_map:
                continue

            li = meta_map[doc_id]
            index_obj = load_index_obj(info["index_path"])

            # récupérer le vecteur doc de façon générique (évite xb)
            dv = np.zeros((index_obj.d,), dtype="float32")
            index_obj.reconstruct(li, dv)
            # normaliser pour être cohérent avec IndexFlatIP + normalize_L2
            dv = dv / (np.linalg.norm(dv) + 1e-12)

            return float(np.dot(qv, dv))

        return None


    st.subheader("Résultats")

#     show_others = st.checkbox(
#     "Afficher scores des autres modèles",
#     key="show_others_scores",
#     value=False
# )

    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        if i < 0 or i >= len(meta):
            continue
        row = meta[i]

        label = (
            row.get("term")
            or row.get("label")
            or row.get("doc_id")
            or f"doc#{rank}"
        )
        st.markdown(f"**{rank}. {label}** — score={s:.4f}")

        cols = st.columns([1, 2])


        doc_id = row.get("doc_id")

        if compare_others and doc_id:
            scores_by_model = []
            # modèle sélectionné (score déjà calculé)
            scores_by_model.append({"model": model_display.get(model_ref, model_ref), "score": float(s)})

            for om in other_models:
                sc = score_doc_for_model(om, doc_id, query)
                if sc is not None:
                    scores_by_model.append({"model": model_display.get(om, om), "score": sc})

            # tri décroissant pour voir qui score le mieux ce doc
            scores_by_model = sorted(scores_by_model, key=lambda x: x["score"], reverse=True)
            st.dataframe(scores_by_model, use_container_width=True, hide_index=True)


        with cols[0]:
            # Afficher seulement si présent
            for k in ["subset", "doc_id", "source_row_id", "k"]:
                if row.get(k) is not None:
                    st.write(f"{k}:", row.get(k))

        with cols[1]:
            # Champs “principaux” si présents
            for k in ["query", "positive"]:
                if row.get(k):
                    st.markdown(f"**{k}**")
                    st.write(row[k])

            # Le reste en meta (toujours dispo)
            with st.expander("meta"):
                st.json(row)

        st.divider()