import torch


@torch.no_grad()
def greedy_generate_until_sep(model, tokenizer, prompt: str, *, max_new_tokens: int) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = int(input_ids.shape[-1])
    for _ in range(max_new_tokens):
        if input_ids.size(-1) > model.config.n_positions:
            input_ids = input_ids[:, -model.config.n_positions :]
        out = model(input_ids=input_ids)
        next_token_logits = out["logits"][:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        if tokenizer.sep_token_id is not None and (next_token_id == tokenizer.sep_token_id).any():
            break
    generated_ids = input_ids[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=False)

