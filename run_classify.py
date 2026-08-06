import argparse
import csv
import json
import os
from typing import Dict, Iterator, Optional

import torch

from lib import load_model_and_tokenizer


def iter_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max_length", type=int, default=13000)
    ap.add_argument("--num_labels", type=int, default=None, help="Optional; inferred from config.json if missing")
    args = ap.parse_args()

    num_labels = args.num_labels
    cfg_path = os.path.join(args.model_dir, "config.json")
    id2label = None
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
        if num_labels is None and isinstance(cfg.get("label2id"), dict):
            num_labels = len(cfg["label2id"])
        if isinstance(cfg.get("id2label"), dict):
            id2label = {int(k): v for k, v in cfg["id2label"].items()}

    if num_labels is None:
        raise SystemExit("num_labels missing and cannot be inferred from config.json (label2id not found)")

    model, tokenizer = load_model_and_tokenizer(args.model_dir, num_labels=num_labels)
    device = model.device

    rows = []
    y_true = []
    y_pred = []

    for ex in iter_jsonl(args.input_jsonl):
        text = ex.get("input")
        true_label = ex.get("label")
        if not isinstance(text, str):
            continue
        enc = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        if "token_type_ids" in enc:
            del enc["token_type_ids"]
        out = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
        logits = out.get("classification_logits")
        probs = torch.softmax(logits, dim=-1)
        max_prob, pred_id = torch.max(probs, dim=-1)
        conf = float(max_prob.item())
        pred_id = int(pred_id.item())
        if conf < args.threshold:
            pred_label = "UNK"
        else:
            pred_label = id2label.get(pred_id, str(pred_id)) if id2label else str(pred_id)

        rows.append({"input": text, "true_label": true_label, "pred_label": pred_label, "confidence": conf})
        if id2label and true_label in set(id2label.values()):
            inv = {v: k for k, v in id2label.items()}
            y_true.append(inv[true_label])
            y_pred.append(pred_id)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["input", "true_label", "pred_label", "confidence"])
        w.writeheader()
        w.writerows(rows)

    acc = None
    if y_true:
        acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    print(os.path.abspath(args.output_csv), "accuracy" if acc is not None else "accuracy=N/A", acc)


if __name__ == "__main__":
    main()

