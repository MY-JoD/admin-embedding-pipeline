import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import faiss
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
DEFAULT_JSONL = "data-0/Dataset_Bercy_4k_lines.jsonl"
LOCAL_DIR = Path("./models_ft/merged")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MAX_LENGTH = 512

TASK = "Retrieve the definition of an administrative acronym or term."
E5_Q_PREFIX = f"Instruct: {TASK}\nQuery: "

MODELS: Dict[str, str] = {
    # E5
    "E5 Base (instruct)": "intfloat/multilingual-e5-large-instruct",
    "E5 FT config2 v2": str(LOCAL_DIR / "e5_large_finetuned_config2_merged_v2"),
    "E5 FT config4 v2": str(LOCAL_DIR / "e5_large_finetuned_config4_merged_v2"),

    # Solon
    "Solon Base": "OrdalieTech/SOLON-embeddings-large-0.1",
    "Solon FT config2 v2": str(LOCAL_DIR / "solon_large_finetuned_config2_merged_v2"),
    "Solon FT config4 v2": str(LOCAL_DIR / "solon_large_finetuned_config4_merged_v2"),
}

# =========================
# HELPERS
# =========================
def norm_space(s: str) -> str:
    return " ".join(str(s).strip().split())

def model_query_prefix(model_name: str) -> str:
    # Important: query prefix differs by model family
    if model_name.startswith("E5"):
        return E5_Q_PREFIX
    if model_name.startswith("Solon"):
        return "query : "
    return ""

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_def_docs(rows: List[dict]) -> Tuple[List[str], List[dict]]:
    """
    Docs = definitions only: pair_type == 'def'
    Keep 1 doc per cluster_id.
    """
    docs: List[str] = []
    meta: List[dict] = []
    seen = set()

    for r in rows:
        if r.get("pair_type") != "def":
            continue
        cid = r.get("cluster_id")
        if not cid or cid in seen:
            continue

        pos = norm_space(r.get("positive", ""))
        if not pos:
            continue

        dom = norm_space(r.get("domain", "UNKNOWN"))
        docs.append(pos)
        meta.append({"cluster_id": cid, "domain": dom})
        seen.add(cid)

    return docs, meta

def encode(model: SentenceTransformer, texts: List[str], prefix: str) -> np.ndarray:
    # normalized embeddings => cosine similarity == inner product
    emb = model.encode(
        [prefix + t for t in texts],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    return emb

def build_faiss_index(doc_emb: np.ndarray) -> faiss.Index:
    dim = doc_emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via inner product (normalized vectors)
    index.add(doc_emb)
    return index

def search(index: faiss.Index, q_emb: np.ndarray, topk: int):
    scores, inds = index.search(q_emb, topk)
    return scores[0], inds[0]  # 1 query

def make_results_df(scores, inds, docs, meta) -> pd.DataFrame:
    rows = []
    for rank, (sc, di) in enumerate(zip(scores.tolist(), inds.tolist()), start=1):
        rows.append({
            "rank": rank,
            "score": round(float(sc), 4),
            "cluster_id": meta[di]["cluster_id"],
            "domain": meta[di]["domain"],
            "definition_snippet": docs[di][:200] + ("..." if len(docs[di]) > 200 else "")
        })
    return pd.DataFrame(rows)

def highlight_row_if_match(df: pd.DataFrame, target_cluster_id: str | None):
    if not target_cluster_id:
        return df.style

    target = target_cluster_id.strip().lower()

    def _style_row(row):
        is_match = str(row.get("cluster_id", "")).strip().lower() == target
        if is_match:
            # Couleur soft + texte un peu plus gras
            return ["color: white; background-color: grey; font-weight: 400;"] * len(row)
        return [""] * len(row)

    return df.style.apply(_style_row, axis=1)

# =========================
# STREAMLIT CACHE
# =========================
@st.cache_resource
def get_model(model_id: str) -> SentenceTransformer:
    # Try to silence the tokenizer regex warning if supported by your transformers version
    try:
        m = SentenceTransformer(
            model_id,
            device=DEVICE,
            model_kwargs={"dtype": torch.float16} if DEVICE == "cuda" else {},
            tokenizer_kwargs={"fix_mistral_regex": True},
        )
    except TypeError:
        m = SentenceTransformer(
            model_id,
            device=DEVICE,
            model_kwargs={"dtype": torch.float16} if DEVICE == "cuda" else {},
        )
    m.max_seq_length = MAX_LENGTH
    return m

@st.cache_resource
def load_kb_defs(jsonl_path: str) -> Tuple[List[str], List[dict]]:
    rows = load_jsonl(Path(jsonl_path))
    docs, meta = build_def_docs(rows)
    return docs, meta

@st.cache_resource
def build_index_for_model(model_name: str, model_id: str, jsonl_path: str):
    docs, meta = load_kb_defs(jsonl_path)
    model = get_model(model_id)

    doc_emb = encode(model, docs, prefix="")  # docs = def => no prefix
    index = build_faiss_index(doc_emb)
    return index, docs, meta

# =========================
# UI
# =========================
st.set_page_config(page_title="Demo Désambiguïsation (Def Retrieval)", layout="wide")

st.title("Demo désambiguïsation — Acronyme/Terme → Définition (Top-K)")
st.caption(
    f"Embeddings device: {DEVICE} | Docs = définitions uniquement (pair_type='def') | Similarité = cosinus (embeddings normalisés)"
)

with st.sidebar:
    st.header("Paramètres")

    jsonl_path = st.text_input("Chemin JSONL", value=DEFAULT_JSONL)
    topk = st.slider("Top-K", min_value=1, max_value=50, value=10, step=1)

    model_a = st.selectbox("Modèle A", list(MODELS.keys()), index=0)
    model_b = st.selectbox("Modèle B", list(MODELS.keys()), index=2)

    st.divider()

query = st.text_input("Requête (acronyme/terme)", value="MOA")
highlight_mode = st.toggle("Surligner si query == cluster_id", value=True)

run = st.button("Chercher", type="primary")

if run:
    q = norm_space(query)
    if not q:
        st.warning("Entre une requête non vide.")
        st.stop()

    model_a_id = MODELS[model_a]
    model_b_id = MODELS[model_b]

    with st.spinner("Chargement modèles + index FAISS (cache si déjà fait)…"):
        index_a, docs_a, meta_a = build_index_for_model(model_a, model_a_id, jsonl_path)
        index_b, docs_b, meta_b = build_index_for_model(model_b, model_b_id, jsonl_path)

    # Encode query with correct prefix for each model
    m_a = get_model(model_a_id)
    m_b = get_model(model_b_id)

    q_emb_a = encode(m_a, [q], prefix=model_query_prefix(model_a))
    q_emb_b = encode(m_b, [q], prefix=model_query_prefix(model_b))

    scores_a, inds_a = search(index_a, q_emb_a, topk=topk)
    scores_b, inds_b = search(index_b, q_emb_b, topk=topk)

    highlight_cid = q if highlight_mode else None

    df_a = make_results_df(scores_a, inds_a, docs_a, meta_a)
    df_b = make_results_df(scores_b, inds_b, docs_b, meta_b)

    highlight_cid = q if highlight_mode else None

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Résultats — {model_a}")
        st.data_editor(
            highlight_row_if_match(df_a, highlight_cid),
            use_container_width=True,
            hide_index=True,
            disabled=True
        )

    with col2:
        st.subheader(f"Résultats — {model_b}")
        st.data_editor(
            highlight_row_if_match(df_b, highlight_cid),
            use_container_width=True,
            hide_index=True,
            disabled=True
        )