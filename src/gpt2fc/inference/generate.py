import tiktoken
import torch

from gpt2fc.config import EOS_TOKEN_ID, get_model_config
from gpt2fc.model import GPTModel, KVCache


def load_finetuned_model(checkpoint_path, model_size, device):
    cfg = get_model_config(model_size)
    model = GPTModel(cfg)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def _sample(logits, temperature, top_k):
    if top_k is not None:
        top_logits, _ = torch.topk(logits, k=top_k)
        min_val = top_logits[:, -1]
        logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

    if temperature > 0.0:
        logits = logits / temperature
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=EOS_TOKEN_ID,
             use_cache=True):
    """Autoregressive decoding: greedy by default, sampling when temperature > 0.

    With use_cache (default) each step feeds only the newest token and attends over
    cached K/V — O(n) per token instead of O(n²). The cached path stops at the
    context limit; the uncached path keeps going with a sliding window.
    """
    if use_cache:
        cache = KVCache()
        with torch.no_grad():
            logits = model(idx[:, -context_size:], kv_cache=cache)
        for _ in range(max_new_tokens):
            next_idx = _sample(logits[:, -1, :], temperature, top_k)
            if eos_id is not None and next_idx.item() == eos_id:
                break
            idx = torch.cat((idx, next_idx), dim=1)
            if cache.size >= context_size:
                break
            with torch.no_grad():
                logits = model(next_idx, kv_cache=cache)
        return idx

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        next_idx = _sample(logits[:, -1, :], temperature, top_k)

        if eos_id is not None and next_idx.item() == eos_id:
            break

        idx = torch.cat((idx, next_idx), dim=1)

    return idx


def run_inference(model, tokenizer, prompt, max_new_tokens, device, temperature=0.0, top_k=None):
    """Encode the prompt, generate a completion, and return only the new text."""
    encoded = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    idx = torch.tensor(encoded).unsqueeze(0).to(device)
    out_idx = generate(
        model, idx,
        max_new_tokens=max_new_tokens,
        context_size=model.context_length,
        temperature=temperature,
        top_k=top_k,
    )
    full_text = tokenizer.decode(out_idx.squeeze(0).tolist())
    return full_text[len(prompt):]
