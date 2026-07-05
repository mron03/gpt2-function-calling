import torch

from gpt2fc.model import GPTModel

TINY_CFG = {
    "vocab_size": 101,
    "context_length": 16,
    "emb_dim": 12,
    "n_heads": 3,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


def test_forward_shape():
    model = GPTModel(TINY_CFG)
    x = torch.randint(0, TINY_CFG["vocab_size"], (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, TINY_CFG["vocab_size"])


def test_context_length_property():
    model = GPTModel(TINY_CFG)
    assert model.context_length == TINY_CFG["context_length"]


def test_causal_masking():
    """Changing a future token must not change logits at earlier positions."""
    torch.manual_seed(0)
    model = GPTModel(TINY_CFG)
    model.eval()

    x = torch.randint(0, TINY_CFG["vocab_size"], (1, 8))
    x_perturbed = x.clone()
    x_perturbed[0, -1] = (x[0, -1] + 1) % TINY_CFG["vocab_size"]

    with torch.no_grad():
        logits = model(x)
        logits_perturbed = model(x_perturbed)

    assert torch.allclose(logits[0, :-1], logits_perturbed[0, :-1], atol=1e-5)
    assert not torch.allclose(logits[0, -1], logits_perturbed[0, -1], atol=1e-5)


def test_parameter_count_tiny():
    """Parameter count matches the analytic formula for the architecture."""
    model = GPTModel(TINY_CFG)
    V, C, E, L = (TINY_CFG["vocab_size"], TINY_CFG["context_length"],
                  TINY_CFG["emb_dim"], TINY_CFG["n_layers"])
    per_block = (
        3 * (E * E + E)      # Q, K, V projections (with bias)
        + E * E + E          # output projection
        + E * (4 * E) + 4 * E + (4 * E) * E + E  # feed-forward
        + 4 * E              # two LayerNorms (scale + shift)
    )
    expected = V * E + C * E + L * per_block + 2 * E + E * V
    assert sum(p.numel() for p in model.parameters()) == expected
