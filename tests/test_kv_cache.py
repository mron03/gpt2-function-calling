import torch

from gpt2fc.inference.generate import generate
from gpt2fc.model import GPTModel, KVCache

TINY_CFG = {
    "vocab_size": 101,
    "context_length": 16,
    "emb_dim": 12,
    "n_heads": 3,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


def make_model(seed=0):
    torch.manual_seed(seed)
    model = GPTModel(TINY_CFG)
    model.eval()
    return model


def test_incremental_forward_matches_full():
    """Priming with a prefix then feeding tokens one at a time must reproduce
    the last-position logits of a full uncached forward pass."""
    model = make_model()
    x = torch.randint(0, TINY_CFG["vocab_size"], (1, 10))

    with torch.no_grad():
        full = model(x)

        cache = KVCache()
        logits = model(x[:, :4], kv_cache=cache)
        assert torch.allclose(logits[:, -1], full[:, 3], atol=1e-5)
        for t in range(4, 10):
            logits = model(x[:, t:t + 1], kv_cache=cache)
            assert torch.allclose(logits[:, -1], full[:, t], atol=1e-5)

    assert cache.size == 10


def test_generate_cached_matches_uncached():
    model = make_model()
    idx = torch.randint(0, TINY_CFG["vocab_size"], (1, 5))
    out_cached = generate(model, idx.clone(), max_new_tokens=8, context_size=16, eos_id=None, use_cache=True)
    out_plain = generate(model, idx.clone(), max_new_tokens=8, context_size=16, eos_id=None, use_cache=False)
    assert torch.equal(out_cached, out_plain)


def test_cached_generation_stops_at_context_limit():
    model = make_model()
    idx = torch.randint(0, TINY_CFG["vocab_size"], (1, 12))
    out = generate(model, idx, max_new_tokens=50, context_size=16, eos_id=None, use_cache=True)
    # 12 prompt positions + 4 generated fill the 16-token context; one more
    # token is sampled from the final logits before the cache runs out.
    assert out.shape[1] <= 12 + 5
