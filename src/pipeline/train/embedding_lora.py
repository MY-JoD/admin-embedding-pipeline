from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

from src.pipeline.model_store import ensure_base_model_local


@dataclass
class TrainConfig:
    base_model_ref: str
    max_seq_length: int = 256
    batch_size: int = 16
    lr: float = 2e-4
    epochs: int = 1
    device: str = "cuda"
    precision: str = "bf16"  # bf16|fp16|fp32


class PairJsonlDataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.rows: List[Dict[str, str]] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.rows.append({"query": obj["query"], "positive": obj["positive"]})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.rows[idx]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _autocast_params(device: torch.device, precision: str):
    if device.type != "cuda":
        return False, torch.float32
    if precision == "fp16":
        return True, torch.float16
    if precision == "bf16":
        return True, torch.bfloat16
    return False, torch.float32


def train_embedding_lora_chain(
    train_jsonl: Path,
    output_adapter_dir: Path,
    base_model_ref: str,
    lora_cfg: Dict,
    cfg: TrainConfig,
    parent_adapter_dir: Optional[Path] = None,
) -> Dict[str, float]:
    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(base_model_ref, use_fast=True)
    model = AutoModel.from_pretrained(base_model_ref)

    # resolved_ref = ensure_base_model_local({"base_model_ref": base_model_ref})
    # tokenizer = AutoTokenizer.from_pretrained(resolved_ref, use_fast=True)
    # model = AutoModel.from_pretrained(resolved_ref)

    # Create or load LoRA
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type="FEATURE_EXTRACTION",
    )

    model = get_peft_model(model, peft_config)

    # If parent adapter exists, load its weights into the LoRA layers (chain)
    if parent_adapter_dir is not None and parent_adapter_dir.exists():
        model = PeftModel.from_pretrained(model, str(parent_adapter_dir), is_trainable=True)

    model.to(device)
    model.train()

    use_amp, amp_dtype = _autocast_params(device, cfg.precision)

    ds = PairJsonlDataset(train_jsonl)

    def collate(batch):
        q = [b["query"] for b in batch]
        p = [b["positive"] for b in batch]
        tq = tokenizer(q, padding=True, truncation=True, max_length=cfg.max_seq_length, return_tensors="pt")
        tp = tokenizer(p, padding=True, truncation=True, max_length=cfg.max_seq_length, return_tensors="pt")
        return tq, tp

    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    total_loss = 0.0
    steps = 0

    for _ in range(cfg.epochs):
        for tq, tp in dl:
            tq = {k: v.to(device) for k, v in tq.items()}
            tp = {k: v.to(device) for k, v in tp.items()}

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                oq = model(**tq).last_hidden_state
                op = model(**tp).last_hidden_state
                eq = mean_pool(oq, tq["attention_mask"])
                ep = mean_pool(op, tp["attention_mask"])

                eq = F.normalize(eq, p=2, dim=1)
                ep = F.normalize(ep, p=2, dim=1)

                logits = eq @ ep.t()
                labels = torch.arange(logits.size(0), device=device)
                loss = F.cross_entropy(logits, labels)

            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu())
            steps += 1

    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_adapter_dir))

    return {"train_loss": total_loss / max(steps, 1), "steps": steps, "num_samples": len(ds)}
