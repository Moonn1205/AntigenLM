import json
import os
from typing import List, Optional

from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import AddedToken


class InfluTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "N",
        pad_token: str = "<pad>",
        eos_token: str = "<eos>",
        sep_token: str = "<sep>",
        additional_special_tokens=None,
        **kwargs,
    ):
        with open(vocab_file, "r") as f:
            self.vocab = json.load(f)
        self.id2token = {v: k for k, v in self.vocab.items()}
        if additional_special_tokens is None:
            additional_special_tokens = ["<HA>", "<NA>"]
        super().__init__(
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        i = 0
        while i < len(text):
            if text[i] == "<":
                j = text.find(">", i)
                if j != -1:
                    tok = text[i : j + 1]
                    if tok in self.vocab:
                        tokens.append(tok)
                        i = j + 1
                        continue
            ch = text[i].upper()
            tokens.append(ch if ch in self.vocab else self.unk_token)
            i += 1
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.id2token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(save_directory, (filename_prefix or "") + "vocab.json")
        with open(vocab_file, "w") as f:
            json.dump(self.vocab, f, indent=2)
        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
        tokenizer_config_file = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
        special_tokens_map_file = os.path.join(pretrained_model_name_or_path, "special_tokens_map.json")
        added_tokens_file = os.path.join(pretrained_model_name_or_path, "added_tokens.json")

        config = {}
        if os.path.exists(tokenizer_config_file):
            with open(tokenizer_config_file) as f:
                config.update(json.load(f))
        if os.path.exists(special_tokens_map_file):
            with open(special_tokens_map_file) as f:
                config.update(json.load(f))

        tok = cls(vocab_file=vocab_file, **config)
        if os.path.exists(added_tokens_file):
            with open(added_tokens_file) as f:
                added_tokens_dict = json.load(f)
            for t in added_tokens_dict:
                tok.add_tokens([AddedToken(t, lstrip=False, rstrip=False)])
        return tok

