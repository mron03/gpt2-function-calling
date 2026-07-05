import torch
import torch.nn as nn

from gpt2fc.model.layers import LayerNorm, TransformerBlock


class LayerKVCache:
    """Cached key/value tensors for one attention layer, shape (b, heads, seq, head_dim)."""

    def __init__(self):
        self.k = None
        self.v = None

    def update(self, keys, values):
        if self.k is not None:
            keys = torch.cat((self.k, keys), dim=2)
            values = torch.cat((self.v, values), dim=2)
        self.k, self.v = keys, values
        return keys, values


class KVCache:
    """Per-layer K/V cache for incremental decoding.

    Pass a fresh instance to GPTModel.forward, then feed only the new token(s)
    on subsequent calls — attention still sees the full sequence.
    """

    def __init__(self):
        self._layers = {}

    def layer(self, i):
        if i not in self._layers:
            self._layers[i] = LayerKVCache()
        return self._layers[i]

    @property
    def size(self):
        first = self._layers.get(0)
        return 0 if first is None or first.k is None else first.k.shape[2]


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    @property
    def context_length(self):
        return self.pos_emb.weight.shape[0]

    def forward(self, in_idx, kv_cache=None):
        batch_size, seq_len = in_idx.shape
        n_past = kv_cache.size if kv_cache is not None else 0
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(n_past, n_past + seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        if kv_cache is None:
            x = self.trf_blocks(x)
        else:
            for i, block in enumerate(self.trf_blocks):
                x = block(x, kv_cache.layer(i))
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
