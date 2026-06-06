"""
Fine-tuning de BETO (BERT en español) para detección de noticias falsas.

Modelo base: dccuchile/bert-base-spanish-wwm-cased  (Cañete et al., 2020)
Requiere GPU para tiempos razonables. Si no hay GPU, el script lo advierte y
puede correrse igual (lento) reduciendo épocas/batch.

Salidas:
  models/beto/                    -> modelo fine-tuneado
  results/metrics_beto.json       -> métricas en el conjunto de prueba
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as data_mod

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models" / "beto"

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LEN = 256
SEED = 42


def _compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    logits, labels = eval_pred
    # softmax para prob de la clase positiva (fake=1)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_fake": f1_score(labels, preds, pos_label=1),
        "roc_auc": auc,
    }


def run(epochs: int = 4, batch_size: int = 16, lr: float = 2e-5) -> dict:
    """Fine-tunea BETO y evalúa en el conjunto de prueba."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    if device == "cpu":
        print("⚠ Sin GPU detectada: el fine-tuning será MUY lento. "
              "Considera reducir --epochs y --batch-size, o usar el pipeline clásico.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    splits = data_mod.load_all()
    df_tr = pd.concat([splits["train"], splits["development"]], ignore_index=True)
    df_te = splits["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    ds_tr = Dataset.from_pandas(df_tr[["text", "label"]]).map(tokenize, batched=True)
    ds_te = Dataset.from_pandas(df_te[["text", "label"]]).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir=str(MODELS),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=20,
        seed=SEED,
        report_to="none",
        fp16=(device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_te,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    print("== Entrenando BETO ==")
    trainer.train()
    metrics = trainer.evaluate()
    print("Métricas finales:", metrics)

    trainer.save_model(str(MODELS))
    tokenizer.save_pretrained(str(MODELS))
    (RESULTS / "metrics_beto.json").write_text(json.dumps(metrics, indent=2))
    print(f"Modelo guardado en {MODELS}")
    return metrics


if __name__ == "__main__":
    run()
