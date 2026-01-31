import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses
)
from torch.utils.data import DataLoader, SequentialSampler
from sentence_transformers.training_args import BatchSamplers
from peft import LoraConfig, TaskType, get_peft_model
import os

# TODO : Install library
# pip install -q sentence-transformers peft datasets torch accelerate transformers


# 1. PARAMÈTRES (CONFIG 4) - Version 2
MODEL_ID = "intfloat/multilingual-e5-large-instruct"

# Nouveaux dossiers de sortie
OUTPUT_DIR_LORA = "adapters/e5_lora_config4_qkv_hn_adapter_v2"
OUTPUT_DIR_MERGED = "final_models/e5_large_finetuned_config4_merged_v2"

# Hyperparamètres
BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 512

# 2. PRÉPARATION ET TRI DES DONNÉES

def prepare_sorted_dataset(filepath):
    print(f"Chargement des données depuis {filepath}...")
    dataset = load_dataset("json", data_files=filepath, split="train")

    # 1. Ajout de l'instruction
    def add_prefixes(example):
        example["anchor"]   = QUERY_PREFIX + example["anchor"]
        example["positive"] = DOC_PREFIX + example["positive"]
        if "negative" in example and example["negative"]:
            example["negative"] = DOC_PREFIX + example["negative"]
        return example
    dataset = dataset.map(add_prefixes)

    seen = set()
    def keep_first(example):
        a = example["anchor"]
        if a in seen:
            return False
        seen.add(a)
        return True

    dataset = dataset.filter(keep_first)

    # 2. TRI PAR CLUSTER (C'est ici que se joue la stratégie Hard Negatives)
    # En triant par 'cluster_id', on regroupe les concepts proches (ex: tous les 'RH' ensemble)
    # Ainsi, un batch contiendra plusieurs exemples du même cluster, rendant la distinction difficile.
    print("--- Stratégie Config 3 : Tri du dataset par similarité (cluster_id) ---")
    dataset = dataset.sort("cluster_id")

    # Vérification visuelle
    print(f"Exemple 0 (Cluster {dataset[0]['cluster_id']}) : {dataset[0]['anchor']}")
    print(f"Exemple 1 (Cluster {dataset[1]['cluster_id']}) : {dataset[1]['anchor']}")

    return dataset

train_dataset = prepare_sorted_dataset("bercy_train_90.jsonl")

# 3. CLASS TRAINER SPÉCIFIQUE (NO SHUFFLE)

class NoShuffleTrainer(SentenceTransformerTrainer):
    """
    Indispensable pour la Config 4 :
    Empêche le mélange aléatoire pour conserver l'effet 'Hard Negatives' du tri.
    """
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=SequentialSampler(self.train_dataset),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# 4. MODÈLE & CONFIGURATION 4 (Q + K + V)

print("Chargement du modèle de base...")
model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
model.max_seq_length = MAX_SEQ_LENGTH
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query", "key", "value"],
)

model[0].auto_model = get_peft_model(model[0].auto_model, peft_config)
print("\n--- Paramètres entraînables (Config 4 : Q+K+V + Hard Negs) ---")
model[0].auto_model.print_trainable_parameters()

# 5. ENTRAÎNEMENT

train_loss = losses.MultipleNegativesRankingLoss(model=model)

args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR_LORA,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.1,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
    gradient_accumulation_steps=1
)

# Utilisation du Trainer NoShuffle
trainer = NoShuffleTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
)

print("\n--- Démarrage du fine-tuning (Config 4) ---")
trainer.train()

# 6. SAUVEGARDE ET FUSION

print("\n--- Sauvegarde et Fusion ---")
model.save_pretrained(OUTPUT_DIR_LORA)
print("\n--- Fusion des poids (Merge) pour Ragas ---")
# Fusion - Merge
model[0].auto_model = model[0].auto_model.merge_and_unload()
os.makedirs(OUTPUT_DIR_MERGED, exist_ok=True)

model[0].auto_model.save_pretrained(OUTPUT_DIR_MERGED, safe_serialization=True)
model.tokenizer.save_pretrained(OUTPUT_DIR_MERGED)
model.save(OUTPUT_DIR_MERGED)

print(f"Modèle Config 4 FUSIONNÉ sauvegardé dans : {OUTPUT_DIR_MERGED}")
