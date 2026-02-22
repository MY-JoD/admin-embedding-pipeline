import os
import json
import hashlib
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import torch

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# CONFIG

EMB_MODEL_PATH = "final_models/e5_large_finetuned_config4_merged_v2"
E5_INSTRUCTION = "Retrieve relevant passages for answering the query."
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "llama3.2:3b-instruct-q4_K_M"
VECTORSTORE_DIR = "vectorstore"

INDEX_VERSION = "v2_url_chunking"
URL_CHUNK_SIZE = 650
URL_CHUNK_OVERLAP = 100

ADMIN_PASSWORD = (
    Path(".env_admin").read_text(encoding="utf-8").strip()
    if Path(".env_admin").exists()
    else os.environ.get("RAG_ADMIN_PASSWORD")
)

# PAGE CONFIG + UI

st.set_page_config(
    page_title="Assistant RAG — Ministère des Finances",
    page_icon="🏛️",
    layout="wide",
)

CSS = """
<style>
/* GLOBAL */
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
hr { border: none; border-top: 1px solid rgba(10,20,40,0.12); margin: 10px 0; }

/* SIDEBAR THEME */
section[data-testid="stSidebar"] { background: #0B1F3A; }
section[data-testid="stSidebar"] * { color: #F3F6FA !important; }
section[data-testid="stSidebar"] .stButton button { background: #1B4DFF !important; border: 0; }
section[data-testid="stSidebar"] .stButton button:hover { background: #163FE0 !important; }

/* CARDS / BADGES */
.card {
  border: 1px solid rgba(10, 20, 40, 0.15);
  border-radius: 14px;
  padding: 14px 14px;
  background: rgba(245, 247, 250, 0.75);
}
.badge {
  display: inline-block;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(10, 20, 40, 0.18);
  margin-right: 6px;
  background: white;
}
.small { font-size: 12px; opacity: 0.85; }

/* UPLOAD CARD */
.upload-card{
  border: 1px dashed rgba(255,255,255,0.35);
  border-radius: 14px;
  padding: 14px;
  background: rgba(255,255,255,0.06);
}
.upload-title{ font-weight: 700; margin-bottom: 6px; }
.upload-hint{ font-size: 12px; opacity: 0.85; margin-bottom: 10px; }
.file-pill{
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.20);
}

/* FILE UPLOADER */
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  margin-top: 8px !important;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  border-radius: 14px !important;
  padding: 10px 12px !important;
  min-height: 54px !important;
  text-align: center !important;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] small { display: none !important; }
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] p { display: none !important; }
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] div {
  display: flex !important; justify-content: center !important; align-items: center !important;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button {
  background: #1B4DFF !important;
  border: none !important;
  color: white !important;
  border-radius: 10px !important;
  padding: 8px 12px !important;
  font-weight: 600 !important;
  margin-left: auto !important;
  margin-right: auto !important;
  display: inline-flex !important;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button:hover { background: #163FE0 !important; }
/* Optional rename Browse files */
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button { font-size: 0 !important; }
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button::after {
  content: "Choisir un fichier" !important;
  font-size: 14px !important;
}
/* TOP BANNER */
/* on style le bloc de colonnes qui CONTIENT l'ancre */
div[data-testid="stHorizontalBlock"]:has(#top-banner-anchor) {
  border: 1px solid rgba(10, 20, 40, 0.16);
  border-radius: 18px;
  background: rgba(245, 247, 250, 0.95);
  box-shadow:
    0 10px 26px rgba(10, 20, 40, 0.12),
    0 2px 6px rgba(10, 20, 40, 0.08);
  padding: 16px 18px;
  margin: 10px 0 16px 0;
  position: relative;
}
div[data-testid="stHorizontalBlock"]:has(#top-banner-anchor)::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 4px;
  border-radius: 18px 18px 0 0;
  background: rgba(27, 77, 255, 0.22);
}
div[data-testid="stHorizontalBlock"]:has(#top-banner-anchor) [data-testid="stTextInput"] input{
  border-radius: 10px !important;
  background: rgba(245, 247, 250, 0.85) !important;
}
div[data-testid="stHorizontalBlock"]:has(#top-banner-anchor) [data-testid="stButton"] button{
  border-radius: 10px !important;
  font-weight: 700 !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.title("🏛️ Assistant RAG — Bercy Hub x ECE Paris")
st.caption("Recherche sémantique (FAISS + reranking) + LLM local (Ollama) — réponse structurée avec sources.")

# Top bar admin login (BANNER)
st.markdown('<div id="top-banner-anchor"></div>', unsafe_allow_html=True)

top_left, top_right = st.columns([0.72, 0.28], vertical_alignment="center")

with top_left:
    st.markdown('<span id="top-banner-anchor"></span>', unsafe_allow_html=True)

    st.markdown("### Accès avancé")
    st.caption("Optionnel — réservé data scientist. Permet d'afficher les passages et de régler les paramètres (MMR, k, rerank, etc.).")

with top_right:
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    if st.session_state.is_admin:
        st.markdown("**Admin**")
        if st.button("Se déconnecter", use_container_width=True):
            st.session_state.is_admin = False
            st.rerun()
    else:
        st.markdown("**Connexion admin**")
        pwd = st.text_input(
            "Mot de passe",
            type="password",
            placeholder="Mot de passe",
            label_visibility="collapsed",
            key="admin_pwd",
        )
        if st.button("Connexion", use_container_width=True):
            if ADMIN_PASSWORD and pwd == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.rerun()
            else:
                st.session_state.is_admin = False
                st.error("Mot de passe incorrect.")

# HELPERS

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

def load_jsonl_as_docs(jsonl_bytes: bytes, source_name: str) -> List[Document]:
    docs = []
    lines = jsonl_bytes.decode("utf-8", errors="ignore").splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        row = json.loads(line)

        pair_type = (row.get("pair_type") or "").lower().strip()
        if pair_type == "qa":
            content = f"Q: {row.get('anchor','')}\nA: {row.get('positive','')}"
        elif pair_type == "def":
            content = f"{row.get('anchor','')} — {row.get('positive','')}"
        else:
            content = row.get("positive", "")

        meta = {
            "source": source_name,
            "line_id": i,
            "cluster_id": row.get("cluster_id"),
            "domain": row.get("domain"),
            "anchor_type": row.get("anchor_type"),
            "pair_type": row.get("pair_type"),
            "anchor": row.get("anchor"),
        }
        docs.append(Document(page_content=content, metadata=meta))
    return docs

def load_single_url_as_docs(url: str, chunk_size: int = URL_CHUNK_SIZE, chunk_overlap: int = URL_CHUNK_OVERLAP) -> List[Document]:
    """Récupère une URL, extrait le texte et le découpe en chunks."""
    if not url:
        return []

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        # retirer éléments non utiles
        for t in soup(["script", "style", "nav", "footer", "header", "aside"]):
            t.decompose()

        text = soup.get_text(separator="\n")
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        text = text.strip()

        # garde-fou
        if len(text) < 400:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)

        docs = []
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": url,
                        "pair_type": "web",
                        "anchor": url,
                        "domain": "Web",
                        "chunk_id": i,
                    },
                )
            )
        return docs

    except Exception:
        return []

def format_sources(source_documents: List[Document], max_chars=600) -> Tuple[str, str]:
    context_parts, sources_parts = [], []
    for idx, doc in enumerate(source_documents, start=1):
        sid = f"S{idx}"
        text = (doc.page_content or "").strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"

        meta = doc.metadata or {}
        pos = meta.get("line_id", meta.get("chunk_id", "NA"))

        sources_parts.append(
            f"[{sid}] {meta.get('source','unknown')} — line {pos} — "
            f"{meta.get('pair_type','NA')} — {meta.get('anchor','NA')} — {meta.get('domain','NA')}"
        )
        context_parts.append(f"[{sid}] {text}")

    return "\n\n".join(context_parts), "\n".join(sources_parts)

def load_pdfs_as_docs(
    pdf_files,
    chunk_size=600,
    chunk_overlap=100
) -> List[Document]:
    docs = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            full_text = ""

            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    full_text += "\n" + txt

            full_text = full_text.strip()
            if len(full_text) < 300:
                continue

            chunks = splitter.split_text(full_text)

            for i, chunk in enumerate(chunks):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf.name,
                            "pair_type": "pdf",
                            "domain": "PDF",
                            "anchor": pdf.name,
                            "chunk_id": i,
                        },
                    )
                )

        except Exception as e:
            print(f"PDF error {pdf.name}: {e}")

    return docs

# EMBEDDINGS

class E5InstructEmbeddings(Embeddings):
    def __init__(self, model_path: str, instruction: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        )
        self.instruction = instruction

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.base.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.base.embed_query(f"{self.instruction}\n{text}")

# PROMPT

PROMPT_TEMPLATE_CITED = """Tu es un assistant rigoureux.

RÈGLES IMPORTANTES:
- Tu dois répondre UNIQUEMENT à partir du CONTEXTE.
- Si la réponse n'est pas dans le contexte, dis: "Je ne sais pas d'après le contexte fourni."
- CITATIONS: utilise UNIQUEMENT les citations disponibles: {allowed_citations}
- N'invente JAMAIS de citations (ex: pas de [S2] si [S2] n'existe pas).
- Réponds en français, clair et professionnel.

CONTEXTE:
{context}

QUESTION:
{question}

Structure attendue:
1) Définition (1-2 phrases) + citations
2) Détails / points clés (2-5 bullet points) + citations
3) Exemple d'usage (1 bullet point) + citation (si possible)
"""
prompt = PromptTemplate(template=PROMPT_TEMPLATE_CITED, input_variables=["context", "question", "allowed_citations"])

# CACHED FAISS BUILD/LOAD

@st.cache_resource(show_spinner=True)
def build_or_load_faiss(corpus_id: str, docs: List[Document]):
    embeddings = E5InstructEmbeddings(EMB_MODEL_PATH, E5_INSTRUCTION)
    vs_dir = Path(VECTORSTORE_DIR) / f"faiss_{corpus_id}"
    vs_dir.parent.mkdir(parents=True, exist_ok=True)

    if vs_dir.exists():
        vs = FAISS.load_local(str(vs_dir), embeddings, allow_dangerous_deserialization=True)
        built = False
    else:
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(str(vs_dir))
        built = True

    return vs, built, str(vs_dir)

def make_retriever(vs: FAISS, use_mmr: bool, k: int, fetch_k: int, lambda_mult: float):
    if use_mmr:
        return vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult})
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

def make_rerank_retriever(base_retriever, top_n: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ce = HuggingFaceCrossEncoder(model_name=RERANK_MODEL, model_kwargs={"device": device})
    compressor = CrossEncoderReranker(model=ce, top_n=top_n)
    return ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)

def rag_answer(question: str, retriever, llm, max_sources: int):
    docs = retriever.invoke(question)[:max_sources]
    context_str, sources_str = format_sources(docs, max_chars=600)

    allowed = " ".join([f"[S{i}]" for i in range(1, len(docs) + 1)]) or "[S1]"
    final_prompt = prompt.format(context=context_str, question=question, allowed_citations=allowed)

    resp = llm.invoke(final_prompt)
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return answer, sources_str, docs

# SESSION STATE

st.session_state.setdefault("ready", False)
st.session_state.setdefault("retriever", None)
st.session_state.setdefault("llm", None)
st.session_state.setdefault("vs_path", "")
st.session_state.setdefault("corpus_info", "")
st.session_state.setdefault("history", [])

# SIDEBAR

with st.sidebar:
    st.divider()
    st.markdown("### 📄 Corpus JSONL")

    st.markdown(
        """
        <div class="upload-card">
          <div class="upload-title">Importer le dataset (.jsonl)</div>
          <div class="upload-hint">Sélectionnez un fichier JSONL pour indexer le corpus (FAISS).</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    jsonl_upload = st.file_uploader(
        "Dataset JSONL",
        type=["jsonl"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="jsonl_uploader",
    )

    if jsonl_upload is not None:
        st.markdown(f"<span class='file-pill'>{jsonl_upload.name} • {round(jsonl_upload.size/1024, 1)} KB</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='file-pill'>Aucun fichier chargé</span>", unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🌐 Source Web (1 URL)")

    use_url = st.toggle("Activer une URL", value=False)
    url_input = st.text_input("URL", placeholder="https://…", disabled=not use_url)

    if use_url and url_input.strip():
        st.caption("✅ URL saisie - elle sera indexée au build.")

    st.divider()

    st.markdown("### 📕 Sources PDF")

    pdf_uploads = st.file_uploader(
        "Importer un ou plusieurs PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    if pdf_uploads:
        for pdf in pdf_uploads:
            st.markdown(
                f"<span class='file-pill'>{pdf.name} • {round(pdf.size/1024,1)} KB</span>",
                unsafe_allow_html=True
            )
    else:
        st.markdown("<span class='file-pill'>Aucun PDF chargé</span>", unsafe_allow_html=True)

    st.info("⚠️ Après ajout ou modification d’une source (JSONL ou URL), relancez « Charger / Construire l’index ».")

    build_btn = st.button(
        "🚀 Charger / Construire l’index",
        use_container_width=True,
        disabled=((jsonl_upload is None) and (not (use_url and url_input.strip()))),
    )

    st.divider()

    if st.session_state.is_admin:
        st.markdown("### ⚙️ Réglages avancés")
        use_mmr = st.toggle("MMR (diversité)", value=False)
        k = st.slider("k (pool initial)", 3, 20, 10)
        fetch_k = st.slider("fetch_k (MMR)", 8, 80, 20, step=2)
        lambda_mult = st.slider("lambda (MMR)", 0.0, 1.0, 0.5, 0.1)

        use_reranker = st.toggle("Activer reranker", value=True)
        top_n = st.slider("top_n (après rerank)", 2, 10, 4)

        temperature = st.slider("Température LLM", 0.0, 1.0, 0.2, 0.05)
        max_sources = st.slider("Sources max au LLM", 2, 10, 4)
    else:
        use_mmr = False
        k = 10
        fetch_k = 20
        lambda_mult = 0.5
        use_reranker = True
        top_n = 4
        temperature = 0.2
        max_sources = 4

    st.divider()
    st.caption("Modèles utilisés :")
    st.caption(f"• Embedding : {EMB_MODEL_PATH}")
    st.caption(f"• Reranker : {RERANK_MODEL}")
    st.caption(f"• LLM : {LLM_MODEL}")

# BUILD / LOAD

if build_btn:
    docs_total = []

    # JSONL (optionnel)
    if jsonl_upload is not None:
        b = jsonl_upload.getvalue()
        docs_jsonl = load_jsonl_as_docs(b, source_name=jsonl_upload.name)
        docs_total.extend(docs_jsonl)
        jsonl_name = jsonl_upload.name
    else:
        b = b""
        jsonl_name = None

    # PDF (optionnel)
    if pdf_uploads:
        docs_pdf = load_pdfs_as_docs(pdf_uploads)
        docs_total.extend(docs_pdf)
    else:
        docs_pdf = []

    # URL (optionnel)
    url_clean = url_input.strip() if (use_url and url_input.strip()) else ""
    if url_clean:
        docs_url = load_single_url_as_docs(url_clean)
        docs_total.extend(docs_url)
    else:
        docs_url = []

    # Garde-fou
    if not docs_total:
        st.warning("Ajoute au moins une source : JSONL, PDF ou URL.")
        st.stop()

    # ID FAISS
    salt = f"{INDEX_VERSION}|pdf|url|jsonl|cs={URL_CHUNK_SIZE}".encode("utf-8")
    cid = sha256_bytes(
        b
        + b"".join([pdf.getvalue() for pdf in pdf_uploads]) if pdf_uploads else b""
        + (url_clean.encode("utf-8") if url_clean else b"")
        + salt
    )

    # Affichage info corpus
    parts = []
    if jsonl_name:
        parts.append(f"{jsonl_name} ({len(docs_jsonl)})")
    if docs_pdf:
        parts.append(f"{len(pdf_uploads)} PDF")
    if url_clean:
        parts.append("URL (1 page)")

    corpus_info = " + ".join(parts)

    with st.status("Indexation / chargement…", expanded=False):
        vs, built, vs_dir = build_or_load_faiss(cid, docs_total)

        base_ret = make_retriever(
            vs,
            use_mmr=use_mmr,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        retriever = make_rerank_retriever(base_ret, top_n) if use_reranker else base_ret

        llm = ChatOllama(
            model=LLM_MODEL,
            temperature=temperature,
            options={"num_ctx": 2048, "num_predict": 256},
        )

    st.session_state.ready = True
    st.session_state.retriever = retriever
    st.session_state.llm = llm
    st.session_state.vs_path = vs_dir
    st.session_state.corpus_info = corpus_info

    st.success(f"✅ Prêt — index {'créé' if built else 'chargé'} : {vs_dir}")

# MAIN LAYOUT

left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("💬 Assistant")
    if st.session_state.corpus_info:
        st.caption(f"Corpus : {st.session_state.corpus_info}  •  Index : {st.session_state.vs_path}")

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    user_q = st.chat_input("Exemple : C'est quoi un RSSI ?")
    if user_q:
        st.session_state.history.append(("user", user_q))
        with st.chat_message("user"):
            st.markdown(user_q)

        if not st.session_state.ready:
            with st.chat_message("assistant"):
                st.warning("Charge le corpus JSONL via la barre latérale puis clique “Charger / Construire l’index”.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Recherche + génération…"):
                    answer, sources_str, used_docs = rag_answer(
                        question=user_q,
                        retriever=st.session_state.retriever,
                        llm=st.session_state.llm,
                        max_sources=max_sources,
                    )
                st.markdown(answer)
                st.session_state.history.append(("assistant", answer))

            st.session_state["last_sources"] = sources_str
            st.session_state["last_docs"] = used_docs

with right:
    st.subheader("📚 Traçabilité")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("La réponse s’appuie sur des passages récupérés. Chaque citation [Sx] renvoie à une source.")
    st.markdown("</div>", unsafe_allow_html=True)

    if "last_sources" in st.session_state:
        # Toujours visible
        with st.expander("Sources (format compact)", expanded=True):
            st.code(st.session_state["last_sources"])

        # Visible seulement admin
        if st.session_state.get("is_admin", False):
            with st.expander("Passages utilisés (admin)", expanded=False):
                for i, d in enumerate(st.session_state["last_docs"], start=1):
                    meta = d.metadata or {}
                    st.markdown(f"**[S{i}]** {meta.get('anchor','(sans titre)')}")
                    st.caption(str({k: meta.get(k) for k in ["pair_type","domain","source","line_id","cluster_id","chunk_id"]}))
                    st.write(d.page_content)
                    st.markdown("---")
    else:
        st.info("Pose une question pour voir les sources.")