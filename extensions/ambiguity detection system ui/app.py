# app.py
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st
import faiss


INDEX_ROOT = Path("indexes")


def list_scopes(index_root: Path = INDEX_ROOT):
    if not index_root.exists():
        return []
    scopes = []
    for d in sorted(index_root.iterdir()):
        if d.is_dir() and d.name in ("merged", "base"):
            scopes.append(d.name)
    return scopes


def list_models(scope: str, index_root: Path = INDEX_ROOT):
    d = index_root / scope
    if not d.exists():
        return []
    return [p.name for p in sorted(d.iterdir()) if p.is_dir()]


def list_datasets(scope: str, model: str, index_root: Path = INDEX_ROOT):
    d = index_root / scope / model
    if not d.exists():
        return []
    return [p.name for p in sorted(d.iterdir()) if p.is_dir()]


def has_index(scope: str, model: str, dataset: str, index_root: Path = INDEX_ROOT):
    d = index_root / scope / model / dataset
    return (d / "faiss.index").exists() and (d / "meta.jsonl").exists()


def has_embeddings(scope: str, model: str, dataset: str, index_root: Path = INDEX_ROOT):
    d = index_root / scope / model / dataset
    return (d / "embeddings.npy").exists()


def load_meta(meta_path: Path):
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas


@st.cache_data(show_spinner=False)
def load_meta_cached(scope: str, model: str, dataset: str):
    meta_path = INDEX_ROOT / scope / model / dataset / "meta.jsonl"
    return load_meta(meta_path)


@st.cache_resource(show_spinner=False)
def load_faiss_index_cached(scope: str, model: str, dataset: str):
    idx_path = INDEX_ROOT / scope / model / dataset / "faiss.index"
    return faiss.read_index(str(idx_path))


@st.cache_resource(show_spinner=False)
def load_embeddings_cached(scope: str, model: str, dataset: str):
    emb_path = INDEX_ROOT / scope / model / dataset / "embeddings.npy"
    return np.load(emb_path).astype(np.float32)


def run_analyze_ambiguity_subprocess(
    dataset: str,
    scopeA: str,
    modelA: str,
    scopeB: str,
    modelB: str,
    k: int,
    t_close: float,
    t_far: float,
    dmin: float,
    selection_mode: str,
    sample: int,
    seed: int,
    start: int,
    count: int,
    max_pairs: int,
    max_items_out: int,
    cache: bool,
):
    cmd = [
        sys.executable,
        "scripts/analyze_ambiguity.py",
        "--dataset",
        dataset,
        "--scopeA",
        scopeA,
        "--modelA",
        modelA,
        "--scopeB",
        scopeB,
        "--modelB",
        modelB,
        "--k",
        str(k),
        "--t_close",
        str(t_close),
        "--t_far",
        str(t_far),
        "--dmin",
        str(dmin),
        "--max_pairs",
        str(max_pairs),
        "--max_items_out",
        str(max_items_out),
    ]

    if selection_mode == "Random":
        cmd += ["--sample", str(sample), "--seed", str(seed)]
    else:
        cmd += ["--start", str(start), "--count", str(count)]

    if cache:
        cmd += ["--cache"]

    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def read_jsonl(path: Path, limit: int = 200):
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def find_latest_run_dir(compare_base_dir: Path):
    if not compare_base_dir.exists():
        return None
    run_dirs = [p for p in compare_base_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0]


def query_single_index(index, meta, query_vec, top_k: int):
    D, I = index.search(query_vec, top_k)
    out = []
    for rank in range(top_k):
        j = int(I[0, rank])
        s = float(D[0, rank])
        if j < 0 or j >= len(meta):
            continue
        out.append({"rank": rank + 1, "score": s, **meta[j]["raw"]})
    return out


st.set_page_config(page_title="PFE - Retrieval / Ambiguity", layout="wide")
st.title("PFE - Retrieval / Ambiguity")

tab_query, tab_amb = st.tabs(["Recherche", "Ambiguïtés"])

# -------------------------
# TAB: Recherche
# -------------------------
with tab_query:
    st.subheader("Recherche top-k dans un index (1 dataset)")
    scopes = list_scopes()
    if not scopes:
        st.error("Aucun dossier indexes/merged ou indexes/base. Lance d'abord le build des index.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        scope = st.selectbox("Scope", scopes, key="search_scope", index=0)
    with col2:
        models = list_models(scope)
        model = st.selectbox("Modèle", models, key="search_model") if models else None
    with col3:
        datasets = list_datasets(scope, model) if model else []
        dataset = st.selectbox("Dataset", datasets, key="search_dataset") if datasets else None

    if not model or not dataset:
        st.info("Sélectionne un modèle et un dataset.")
        st.stop()

    ok = has_index(scope, model, dataset) and has_embeddings(scope, model, dataset)
    if not ok:
        st.warning("Index/embeddings non présents pour ce modèle+dataset. Lance l'indexation.")
        st.stop()

    top_k = st.slider("Top-k", 1, 50, 10, key="search_topk")
    i_query = st.number_input(
        "Focus item index i (optionnel, -1 = désactivé)",
        min_value=-1,
        max_value=10_000_000,
        value=-1,
        step=1,
        key="search_i",
    )

    st.caption("Mode rapide: pas d'encodage texte ici. Tu fournis i -> on cherche ses voisins dans l'index.")

    if st.button("Chercher voisins", key="search_btn"):
        index = load_faiss_index_cached(scope, model, dataset)
        meta = load_meta_cached(scope, model, dataset)
        emb = load_embeddings_cached(scope, model, dataset)

        if i_query < 0 or i_query >= emb.shape[0]:
            st.error("i invalide (hors range).")
        else:
            q = emb[int(i_query) : int(i_query) + 1]
            rows = query_single_index(index, meta, q, top_k)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# -------------------------
# TAB: Ambiguïtés
# -------------------------
with tab_amb:
    st.subheader("Analyse d'ambiguïtés entre 2 modèles (1 dataset)")
    scopes = list_scopes()
    if not scopes:
        st.error("Aucun dossier indexes/merged ou indexes/base.")
        st.stop()

    colA, colB, colC = st.columns(3)

    with colA:
        ds_set = set()
        for sc in scopes:
            for m in list_models(sc):
                for ds in list_datasets(sc, m):
                    ds_set.add(ds)

        dataset_choices = sorted(ds_set)
        dataset = st.selectbox("Dataset", dataset_choices, key="amb_dataset") if dataset_choices else None

    with colB:
        scopeA = st.selectbox("Scope A", scopes, key="amb_scopeA", index=0)
        modelsA = list_models(scopeA)
        modelA = st.selectbox("Modèle A", modelsA, key="amb_modelA") if modelsA else None

    with colC:
        scopeB = st.selectbox("Scope B", scopes, key="amb_scopeB", index=1 if len(scopes) > 1 else 0)
        modelsB = list_models(scopeB)
        modelB = st.selectbox("Modèle B", modelsB, key="amb_modelB") if modelsB else None

    if not dataset or not modelA or not modelB:
        st.info("Sélectionne dataset + 2 modèles.")
        st.stop()

    readyA = has_index(scopeA, modelA, dataset) and has_embeddings(scopeA, modelA, dataset)
    readyB = has_index(scopeB, modelB, dataset) and has_embeddings(scopeB, modelB, dataset)
    if not readyA or not readyB:
        st.warning("Encodages/indices manquants pour une des sélections. Lance build_indexes.py avant.")
        st.stop()

    colp1, colp2, colp3, colp4 = st.columns(4)
    with colp1:
        k = st.slider("k voisins", 2, 50, 10, key="amb_k")
        max_pairs = st.slider("max_pairs", 50, 5000, 500, key="amb_max_pairs")
    with colp2:
        t_close = st.slider("t_close (proche dans A)", 0.0, 1.0, 0.70, key="amb_tclose")
        t_far = st.slider("t_far (loin dans B)", 0.0, 1.0, 0.50, key="amb_tfar")
    with colp3:
        dmin = st.slider("dmin (baisse min)", 0.0, 1.0, 0.15, key="amb_dmin")
        max_items_out = st.slider("max_items_out", 10, 1000, 200, key="amb_max_items_out")
    with colp4:
        selection_mode = st.radio("Sélection items", ["Range", "Random"], index=0, key="amb_selmode")
        use_cache = st.checkbox("Utiliser cache si existe", value=True, key="amb_cache")

    if selection_mode == "Random":
        sample = st.number_input("sample", min_value=1, max_value=1_000_000, value=200, step=10, key="amb_sample")
        seed = st.number_input("seed", min_value=0, max_value=10**9, value=42, step=1, key="amb_seed")
        start = 0
        count = 0
    else:
        start = st.number_input("start", min_value=0, max_value=1_000_000, value=0, step=10, key="amb_start")
        count = st.number_input("count", min_value=1, max_value=1_000_000, value=200, step=10, key="amb_count")
        sample = 0
        seed = 42

    colrun1, colrun2 = st.columns([1, 2])
    with colrun1:
        run = st.button("Lancer analyse", key="amb_run_btn")
    with colrun2:
        st.caption("Le script écrit dans indexes/compare/<dataset>/<A>__vs__<B>/... puis l'UI lit les JSONL.")

    if run:
        with st.spinner("Analyse en cours..."):
            code, out, err = run_analyze_ambiguity_subprocess(
                dataset=dataset,
                scopeA=scopeA,
                modelA=modelA,
                scopeB=scopeB,
                modelB=modelB,
                k=k,
                t_close=t_close,
                t_far=t_far,
                dmin=dmin,
                selection_mode=selection_mode,
                sample=int(sample),
                seed=int(seed),
                start=int(start),
                count=int(count),
                max_pairs=int(max_pairs),
                max_items_out=int(max_items_out),
                cache=use_cache,
            )
        if code != 0:
            st.error("Erreur pendant l'analyse.")
            st.code(err)
        else:
            st.success("OK")
            if out.strip():
                st.code(out)

    compare_base = INDEX_ROOT / "compare" / dataset / f"{scopeA}-{modelA}__vs__{scopeB}-{modelB}"
    run_dir = find_latest_run_dir(compare_base)

    if run_dir is None:
        st.info("Pas encore de résultats. Clique sur 'Lancer analyse'.")
        st.stop()

    st.caption(f"Résultats chargés depuis: {run_dir}")

    dis = read_jsonl(run_dir / "disambiguated.jsonl", limit=500)
    amb = read_jsonl(run_dir / "more_ambiguous.jsonl", limit=500)
    mA = read_jsonl(run_dir / "most_ambiguous_A.jsonl", limit=500)
    mB = read_jsonl(run_dir / "most_ambiguous_B.jsonl", limit=500)

    t1, t2, t3, t4 = st.tabs(["Désambiguïsées", "Devenues ambiguës", "Most ambiguous A", "Most ambiguous B"])

    with t1:
        st.dataframe(pd.DataFrame(dis), use_container_width=True)
    with t2:
        st.dataframe(pd.DataFrame(amb), use_container_width=True)
    with t3:
        st.dataframe(pd.DataFrame(mA), use_container_width=True)
    with t4:
        st.dataframe(pd.DataFrame(mB), use_container_width=True)