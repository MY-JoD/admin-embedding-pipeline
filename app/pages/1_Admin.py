from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import streamlit as st
import yaml
import sys

st.set_page_config(page_title="Admin", layout="wide")
st.title("Admin")

ROOT = Path(".")
CONFIGS_DIR = ROOT / "configs"
EXPS_DIR = ROOT / "experiments"


def list_exp_ids() -> List[str]:
    out = []
    if EXPS_DIR.exists():
        for p in EXPS_DIR.iterdir():
            if p.is_dir():
                out.append(p.name)
    return sorted(out)


def guess_next_split(exp_id: str, asset_id: str) -> str:
    ui_dir = EXPS_DIR / exp_id / "assets_ui" / asset_id / "ui"
    if not ui_dir.exists():
        return "initial"
    splits = []
    for p in ui_dir.glob("*.rich.jsonl"):
        splits.append(p.stem)  # initial, iter01 ...
    if "initial" not in splits:
        return "initial"
    iters = []
    for s in splits:
        m = re.match(r"iter(\d+)$", s)
        if m:
            iters.append(int(m.group(1)))
    nxt = (max(iters) + 1) if iters else 1
    return f"iter{nxt:02d}"


def incoming_has_batch(exp_id: str) -> bool:
    cfg_path = CONFIGS_DIR / f"{exp_id}.yaml"
    incoming = ROOT / f"data_sources/{exp_id}/incoming/batch.jsonl"
    # fallback: même si cfg n'existe pas, on garde le chemin conventionnel
    return incoming.exists()


def validate_yaml_minimal(cfg_text: str) -> Optional[str]:
    try:
        cfg = yaml.safe_load(cfg_text)
    except Exception as e:
        return f"YAML invalide: {e}"

    required = [
        ("exp_id",),
        ("model", "base_model_ref"),
        ("task", "type"),
        ("task", "format"),
        ("task", "text_fields", "query"),
        ("task", "text_fields", "positive"),
        ("data", "incoming_dir"),
        ("ui", "asset_dir"),
        ("train", "enabled"),
    ]
    for path in required:
        cur = cfg
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return f"Config invalide: champ manquant: {'.'.join(path)}"
            cur = cur[k]
    return None


left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("1) Sélection expérience / config")

    mode = st.radio("Mode", ["Sélectionner", "Créer nouvelle"], horizontal=True)

    exp_ids = list_exp_ids()

    if mode == "Sélectionner":
        exp_id = st.selectbox("Expérience (exp_id)", options=exp_ids, index=0 if exp_ids else None)
    else:
        exp_id = st.text_input("Nouvel exp_id", value="")
        exp_id = exp_id.strip()
        if exp_id and not re.match(r"^[A-Za-z0-9_\-]+$", exp_id):
            st.error("exp_id invalide (caractères autorisés: lettres/chiffres/_/-)")
            exp_id = ""


    st.caption("Optionnel : uploader une nouvelle config YAML pour cette expérience (elle écrase configs/<exp_id>.yaml).")
    cfg_upload = st.file_uploader("Config YAML", type=["yaml", "yml"], accept_multiple_files=False)

    cfg_preview = None
    cfg_error = None
    if cfg_upload is not None:
        cfg_preview = cfg_upload.getvalue().decode("utf-8", errors="replace")
        cfg_error = validate_yaml_minimal(cfg_preview)
        if cfg_error:
            st.error(cfg_error)
        else:
            st.success("YAML valide (minimum OK)")

            with st.expander("Voir le YAML"):
                st.code(cfg_preview, language="yaml")
    
    st.markdown("---")
    st.subheader("Enregistrer la config")

    save_btn = st.button("Enregistrer ma config", use_container_width=True)

    if save_btn:
        if not exp_id:
            st.error("exp_id manquant.")
            st.stop()
        if cfg_upload is None:
            st.error("Aucun fichier YAML uploadé.")
            st.stop()
        if cfg_error:
            st.error("YAML invalide, corrige avant d’enregistrer.")
            st.stop()

        # écrire configs/<exp_id>.yaml
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        target = CONFIGS_DIR / f"{exp_id}.yaml"
        target.write_bytes(cfg_upload.getvalue())

        # créer experiments/<exp_id>/
        (EXPS_DIR / exp_id).mkdir(parents=True, exist_ok=True)

        st.success(f"Config enregistrée: {target}")

with right:
    st.subheader("2) Dataset + paramètres")
    dataset = st.file_uploader("Dataset (.jsonl ou .json)", type=["jsonl", "json"], accept_multiple_files=False)

    asset_id = st.text_input("asset_id (namespace UI)", value="sim_0001_term_to_def")

    split_default = "initial"
    if exp_id and asset_id:
        split_default = guess_next_split(exp_id, asset_id)

    split = st.text_input("split", value=split_default, help="initial | iter01 | iter02 ...")

    st.markdown("### Schema (UI)")
    c1, c2 = st.columns(2)
    with c1:
        query_field = st.text_input("query_field", value="query")
        positive_field = st.text_input("positive_field", value="positive")
    with c2:
        label_field = st.text_input("label_field", value="term")
        id_field = st.text_input("id_field", value="source_row_id")

    st.markdown("### Conflit incoming")
    has_incoming = bool(exp_id) and incoming_has_batch(exp_id)
    if has_incoming:
        st.warning("incoming/batch.jsonl existe déjà pour cette expérience.")
    force = st.checkbox("Force (remplacer incoming si déjà présent)", value=has_incoming)

    st.markdown("### Indexing (CPU safe)")
    idx_batch = st.number_input("index batch size", min_value=1, max_value=512, value=16, step=1)
    idx_maxlen = st.number_input("index max length", min_value=16, max_value=2048, value=256, step=16)

st.divider()
st.subheader("Enregistrer la config")
save_cfg = st.button("Enregistrer ma config", key="save_config_btn", use_container_width=True, disabled=(cfg_upload is None or cfg_error is not None))

if save_cfg:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    (EXPS_DIR / exp_id).mkdir(parents=True, exist_ok=True)

    dst = CONFIGS_DIR / f"{exp_id}.yaml"
    existed = dst.exists()

    dst.write_bytes(cfg_upload.getvalue())

    if existed:
        st.success(f"Config mise à jour: {dst}")
    else:
        st.success(f"Config enregistrée: {dst}")

st.subheader("3) Lancer pipeline")
st.caption("Le bouton lance : prepare_iteration -> orchestrator(train) -> build_indexes, et stream les logs ici.")

run_btn = st.button("Lancer", type="primary", use_container_width=True)

log_box = st.empty()

if run_btn:
    if not exp_id:
        st.error("Aucune expérience sélectionnée.")
        st.stop()
    if dataset is None:
        st.error("Dataset manquant.")
        st.stop()
    if cfg_upload is not None and cfg_error:
        st.error("Config YAML invalide.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # dataset temp
        ds_path = tmpdir / dataset.name
        ds_path.write_bytes(dataset.getvalue())

        # config upload temp
        cfg_arg = []
        if cfg_upload is not None:
            cfg_path = tmpdir / cfg_upload.name
            cfg_path.write_bytes(cfg_upload.getvalue())
            cfg_arg = ["--config-upload", str(cfg_path)]

        admin_script = (ROOT / "tools" / "admin" / "admin_launch.py").resolve()

        cmd = [
            sys.executable, str(admin_script),
            "--exp-id", exp_id,
            *cfg_arg,
            "--dataset", str(ds_path),
            "--asset-id", asset_id,
            "--split", split,
            "--query-field", query_field,
            "--positive-field", positive_field,
            "--label-field", label_field,
            "--id-field", id_field,
            "--index-device", "cpu",
            "--index-precision", "fp32",
            "--index-batch-size", str(int(idx_batch)),
            "--index-max-length", str(int(idx_maxlen)),
        ]
        if force:
            cmd.append("--force")


        st.info("Execution en cours...")
        st.code(" ".join(cmd))

        # stream logs
        out_lines = []
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),   # important
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        for line in proc.stdout:
            out_lines.append(line.rstrip("\n"))
            log_box.code("\n".join(out_lines[-300:]), language="text")  # garde les 300 dernières lignes

        rc = proc.wait()
        if rc == 0:
            st.success("OK")
        else:
            st.error(f"Échec (code={rc}). Voir logs.")