import os
import shutil
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download

# Windows: éviter WinError 1314
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


REQUIRED_FILES_ANY = [
    "config.json",
]
REQUIRED_TOKENIZER_ONE_OF = [
    "tokenizer.json",
    "tokenizer_config.json",
]
REQUIRED_WEIGHTS_ONE_OF = [
    "model.safetensors",
    "pytorch_model.bin",
]


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_complete_model_dir(d: Path) -> bool:
    if not d.exists():
        return False

    # required any
    for f in REQUIRED_FILES_ANY:
        if not (d / f).exists():
            return False

    # required tokenizer one-of
    if not any((d / f).exists() for f in REQUIRED_TOKENIZER_ONE_OF):
        return False

    # required weights one-of
    if not any((d / f).exists() for f in REQUIRED_WEIGHTS_ONE_OF):
        return False

    return True


def safe_rmtree(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def download_one(repo_id: str, target_dir: Path, revision: str | None, token: str | None):
    # Si déjà complet -> skip
    if is_complete_model_dir(target_dir):
        print(f"[SKIP] complete: {target_dir}")
        return

    # Si dossier existe mais incomplet -> supprimer
    if target_dir.exists():
        print(f"[WARN] incomplete dir, deleting: {target_dir}")
        safe_rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = [
        "*.json",
        "*.txt",
        "*.md",
        "*.safetensors",
        "*.bin",
        "modules.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "merges.txt",
        "1_Pooling/*",
        "2_Normalize/*",
    ]

    # Téléchargement DIRECT dans target_dir (pas de copie depuis cache)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,   # clé pour Windows
        allow_patterns=allow_patterns,
    )

    # Vérification finale
    if not is_complete_model_dir(target_dir):
        raise RuntimeError(f"Download finished but model dir still incomplete: {target_dir}")

    print(f"[OK] {repo_id}@{revision or 'main'} -> {target_dir}")


def main():
    cfg = load_yaml("config/models.yaml")
    storage_root = Path(cfg.get("storage_root", "models/base"))
    storage_root.mkdir(parents=True, exist_ok=True)

    for m in cfg.get("models", []):
        repo_id = m["repo_id"]
        revision = m.get("revision")
        local_dir = m.get("local_dir") or m["id"]
        token = m.get("token") or os.environ.get("HF_TOKEN")

        target_dir = storage_root / local_dir
        download_one(repo_id, target_dir, revision, token)


if __name__ == "__main__":
    main()