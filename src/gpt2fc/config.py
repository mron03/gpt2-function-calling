import torch

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

MODEL_CONFIGS = {
    "124M":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "355M":  {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "774M":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "1558M": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

EOS_TOKEN_ID = 50256


def get_model_config(model_size):
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size {model_size!r}. Choose from {list(MODEL_CONFIGS)}")
    return {**BASE_CONFIG, **MODEL_CONFIGS[model_size]}


def get_device(preference="auto"):
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
