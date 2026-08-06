import argparse
import json
import os
from typing import Dict, Iterator

from lib import greedy_generate_until_sep, load_model_and_tokenizer


def iter_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=5000)
    args = ap.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for ex in iter_jsonl(args.input_jsonl):
            prompt = ex.get("input")
            pred = greedy_generate_until_sep(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
            out.write(
                json.dumps(
                    {
                        "input": prompt,
                        "prediction": pred,
                        "contains_sep": "<sep>" in pred,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(os.path.abspath(args.output_jsonl))


if __name__ == "__main__":
    main()

