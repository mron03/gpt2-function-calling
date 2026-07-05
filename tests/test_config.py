import pytest
import torch

from gpt2fc.config import MODEL_CONFIGS, get_device, get_model_config


def test_get_model_config_merges_base():
    cfg = get_model_config("124M")
    assert cfg["emb_dim"] == 768
    assert cfg["vocab_size"] == 50257
    assert cfg["context_length"] == 1024


def test_get_model_config_unknown_size():
    with pytest.raises(ValueError):
        get_model_config("13B")


def test_all_sizes_have_head_divisible_dims():
    for size, cfg in MODEL_CONFIGS.items():
        assert cfg["emb_dim"] % cfg["n_heads"] == 0, size


def test_get_device_override():
    assert get_device("cpu") == torch.device("cpu")
