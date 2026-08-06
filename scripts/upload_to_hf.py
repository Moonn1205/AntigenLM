"""Upload AntigenLM model folders to the HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxx              # write-scope token from https://huggingface.co/settings/tokens
    python scripts/upload_to_hf.py                              # uploads both repos
    python scripts/upload_to_hf.py --task generate              # only prediction_sequence
    python scripts/upload_to_hf.py --task classify              # only subtype_classifier
    python scripts/upload_to_hf.py --private                    # create as private
    python scripts/upload_to_hf.py --dry-run                    # list files only, no upload

Excludes training_args.bin / __pycache__ / .DS_Store. Writes a minimal model card
README.md to each Hub repo (does not modify the local repository README).
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login

REPO_ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "generate": {
        "local_dir": REPO_ROOT / "prediction_sequence",
        "repo_id": "Moonn1205/AntigenLM-prediction-sequence",
        "title": "AntigenLM - HA/NA Sequence Forecasting",
        "summary": (
            "Generative head of AntigenLM that autoregressively forecasts the next "
            "time-window influenza HA/NA nucleotide block from three historical blocks."
        ),
    },
    "classify": {
        "local_dir": REPO_ROOT / "subtype_classifier",
        "repo_id": "Moonn1205/AntigenLM-subtype-classifier",
        "title": "AntigenLM - Influenza Subtype Classifier",
        "summary": (
            "Subtype classification head of AntigenLM. Given an `<HA>...<NA>...<sep>` "
            "input, predicts one of 12 influenza A subtypes (H1N1, H3N2, H5N1, ...)."
        ),
    },
}

IGNORE_PATTERNS = ["training_args.bin", "__pycache__/*", ".DS_Store", "*.pyc", "*.log"]


def model_card(task_key: str) -> str:
    task = TASKS[task_key]
    return f"""---
license: mit
library_name: transformers
tags:
  - influenza
  - dna
  - genomics
  - antigen
  - language-model
---

# {task["title"]}

{task["summary"]}

Paper: **AntigenLM: Structure-Aware DNA Language Modeling for Influenza** (arXiv:2602.09067).
Code: https://github.com/Moonn1205/AntigenLM  <!-- update with the real repo URL -->

## Usage

```bash
pip install huggingface_hub transformers==4.29.2 torch==1.13.1
```

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="{task["repo_id"]}",
    local_dir="{task["local_dir"].name}",
    local_dir_use_symlinks=False,
)
```

Then follow the instructions in the
[code repository](https://github.com/Moonn1205/AntigenLM) to run inference.

## Files

- `pytorch_model.bin`  - model weights
- `config.json`        - GPT-2 backbone config + (classifier only) `label2id` / `id2label`
- `vocab.json`, `tokenizer_config.json`, `special_tokens_map.json`{', `added_tokens.json`' if task_key == 'generate' else ''} - tokenizer
"""


def upload_one(task_key: str, api: HfApi, private: bool, dry_run: bool) -> None:
    task = TASKS[task_key]
    local_dir = task["local_dir"]
    repo_id = task["repo_id"]

    if not local_dir.is_dir():
        sys.exit(f"[ERROR] local model directory not found: {local_dir}")

    print(f"\n=== {repo_id}  (<-  {local_dir})")

    files = []
    for p in sorted(local_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir).as_posix()
        skip = any(
            rel == pat or rel.startswith(pat.rstrip("/*") + "/") or rel.endswith(pat.lstrip("*"))
            for pat in IGNORE_PATTERNS
        )
        size_mb = p.stat().st_size / (1024 * 1024)
        flag = "SKIP" if skip else "UP  "
        files.append((flag, rel, size_mb))
        print(f"  [{flag}] {rel:40s} {size_mb:8.2f} MB")

    if dry_run:
        print("  (dry-run; not uploading)")
        return

    print(f"  -> create_repo (exist_ok=True, private={private}) ...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

    card_path = local_dir / "_hf_README.md"
    card_path.write_text(model_card(task_key), encoding="utf-8")
    try:
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
    finally:
        card_path.unlink(missing_ok=True)

    print(f"  -> upload_folder ...")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Upload AntigenLM weights",
    )
    print(f"  done. View at: https://huggingface.co/{repo_id}")
    print(
        f"  verify with: python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id='{repo_id}', local_dir='/tmp/{repo_id.split('/')[-1]}', "
        f"local_dir_use_symlinks=False)\""
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    ap.add_argument("--private", action="store_true", help="create repo as private (default: public)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    if not args.dry_run:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            login(token=token, add_to_git_credential=False)
        try:
            whoami = api.whoami()
            print(f"Logged in as: {whoami.get('name')} ({whoami.get('type')})")
        except Exception as e:
            sys.exit(
                "[ERROR] not logged in to HuggingFace.\n"
                "        either `export HF_TOKEN=hf_xxx` (write scope) "
                "or run `huggingface-cli login` first.\n"
                f"        ({e})"
            )

    targets = list(TASKS) if args.task == "all" else [args.task]
    for t in targets:
        upload_one(t, api, private=args.private, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
