import torch
import torch.nn as nn

from gpt2fc.inference.generate import generate, run_inference

VOCAB = 600


class ScriptedModel(nn.Module):
    """Emits a fixed token sequence regardless of input, then EOS-like token."""

    def __init__(self, script, vocab_size=VOCAB, context_length=32):
        super().__init__()
        self.script = script
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.calls = 0

    def forward(self, idx):
        b, seq_len = idx.shape
        logits = torch.zeros(b, seq_len, self.vocab_size)
        next_token = self.script[min(self.calls, len(self.script) - 1)]
        logits[:, -1, next_token] = 10.0
        self.calls += 1
        return logits


class CharTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_greedy_generation_follows_argmax():
    script = [ord(c) for c in "abc"]
    model = ScriptedModel(script)
    idx = torch.tensor([[1, 2, 3]])
    out = generate(model, idx, max_new_tokens=3, context_size=32, eos_id=None)
    assert out.squeeze(0).tolist() == [1, 2, 3] + script


def test_generation_stops_at_eos():
    eos = 599
    script = [ord("a"), eos, ord("b")]
    model = ScriptedModel(script)
    idx = torch.tensor([[1]])
    out = generate(model, idx, max_new_tokens=10, context_size=32, eos_id=eos)
    # stops before appending EOS; "b" is never generated
    assert out.squeeze(0).tolist() == [1, ord("a")]


def test_run_inference_strips_prompt():
    script = [ord(c) for c in "OK"]
    model = ScriptedModel(script)
    completion = run_inference(model, CharTokenizer(), "hello", max_new_tokens=2, device="cpu")
    assert completion == "OK"
