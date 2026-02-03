import streamlit as st

st.set_page_config(
    page_title="Embedding Incremental Lab",
    layout="wide",
)

st.title("Embedding Incremental Lab")

st.markdown(
    """
Cette app contient deux espaces indépendants :
- **Admin** : uploader une itération (dataset), lancer train + index, suivre les logs
- **Inference** : interroger les indexes d'une expérience / d'un modèle / d'un subset
"""
)

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Admin")
    st.write("Préparer une itération et lancer le pipeline (train + build indexes).")
    st.page_link("pages/1_Admin.py", label="Ouvrir Admin", icon="🛠️")

with c2:
    st.subheader("Inference")
    st.write("Tester / comparer les modèles sur un subset UI indexé.")
    st.page_link("pages/2_Inference.py", label="Ouvrir Inference", icon="🔎")


with c3:
    st.subheader("Test")
    st.write("Tester / comparer les modèles avant/après Fine Tuning.")
    st.page_link("pages/3_Test.py", label="Ouvrir Test", icon="🔎")

