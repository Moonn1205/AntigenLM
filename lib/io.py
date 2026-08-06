import os
from typing import Optional, Tuple

import torch
from transformers import GPT2Config

from .model import GPTForFluMultiTask
from .tokenizer import InfluTokenizer


def _is_git_lfs_pointer(path: str) -> bool:
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(200)
        return head.startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def load_model_and_tokenizer(
    model_dir: str,
    *,
    num_labels: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[GPTForFluMultiTask, InfluTokenizer]:
    tok = InfluTokenizer.from_pretrained(model_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = GPT2Config.from_pretrained(model_dir)
    if num_labels is None:
        num_labels = 0
    model = GPTForFluMultiTask(cfg, num_labels=num_labels)

    weight_path = os.path.join(model_dir, "pytorch_model.bin")
    if _is_git_lfs_pointer(weight_path):
        raise RuntimeError(f"Git LFS pointer detected: {weight_path}")
    sd = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)

    model.eval()
    model.to(device)
    return model, tok

