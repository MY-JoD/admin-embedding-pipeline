# PFE — Automatisation du Fine-Tuning Incrémental de Modèles d’Embeddings (avec Interfaces de Test)

Ce dépôt correspond à un **Projet de Fin d’Études (PFE)** dont l’objectif principal est de mettre en place un système complet permettant **d’automatiser le fine-tuning incrémental de modèles d’embeddings**, puis de **valider** et **analyser** les résultats via des interfaces Streamlit.

Le projet couvre :
- un pipeline de fine-tuning incrémental (itérations successives)
- la gestion des modèles (base / fine-tunés / fusionnés)
- la génération d’index vectoriels (FAISS) pour l’évaluation
- des interfaces de test pour observer l’impact du fine-tuning (retrieval, ambiguïtés, cas d’usage RAG)

---

## Structure du projet

```text
PFE/
├── app/                               # application principale (UI globale)
├── configs/                           # configs globales du projet
├── data_sources/
├── data-0/
├── experiments/                       # expériences / itérations
├── extensions/
│   ├── ambiguity detection system ui/
│   │   ├── config/                    # configs du module (indexing / models)
│   │   ├── data-0/
│   │   ├── datasets/                  # datasets .jsonl (si utilisés localement)
│   │   ├── indexes/                   # index générés (souvent non versionnés)
│   │   ├── scripts/                   # scripts (download models / build indexes / analyze)
│   │   └── app.py                     # UI Streamlit ambiguïtés + retrieval
│   └── UC_rag_interface/
│       ├── app_ministere_rag_v5.py    # UI Streamlit RAG
│       ├── requirements_UC_RAG.txt
│       ├── Dataset_Bercy_4k_lines.jsonl
│       └── Pdf_Ministere.pdf
├── model_staging/
├── models_ft/                         # modèles fine-tunés / sorties
├── registry/
├── src/
├── staging_data/
├── tools/
├── requirements2.txt                  # dépendances globales
└── .gitignore

-----------------------------

## Installation et lancement (module principal)

### 1) Cloner le dépôt
```bash
git clone <repo_url>
cd PFE