import torch

from gpt2fc.training.data import InstructionDataset, custom_collate_fn, format_entry, split_glaive


class StubTokenizer:
    """Character-level stand-in for tiktoken (no network needed)."""

    def encode(self, text, **kwargs):
        return [ord(c) % 500 for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


GLAIVE_ENTRY = {
    "system": "SYSTEM: You are a helpful assistant with access to functions.",
    "chat": (
        "USER: What's the weather in Paris? "
        "\n\n\nASSISTANT: <functioncall> {\"name\": \"get_weather\"} <|endoftext|>"
        "\n\n\nFUNCTION RESPONSE: {\"temp\": 20}"
        "\n\n\nASSISTANT: It is 20 degrees. <|endoftext|>"
    ),
}


def test_format_entry_basic():
    instruction, response = format_entry(GLAIVE_ENTRY)
    assert instruction.startswith("###SYSTEM:")
    assert "###USER: What's the weather in Paris?" in instruction
    assert response.startswith("###ASSISTANT:")
    assert "<functioncall>" in response
    # only the FIRST assistant turn is the response, and it is not in the prompt
    assert "20 degrees" not in response
    assert "###ASSISTANT:" not in instruction
    assert "<|endoftext|>" not in response


def test_format_entry_no_assistant_turn():
    entry = {"system": "SYSTEM: x", "chat": "USER: hello"}
    _, response = format_entry(entry)
    assert response == ""


def test_instruction_dataset_skips_and_filters():
    long_chat = {"system": "SYSTEM: x", "chat": "USER: " + "a" * 5000 + "\n\n\nASSISTANT: hi"}
    no_response = {"system": "SYSTEM: x", "chat": "USER: hello"}
    data = [GLAIVE_ENTRY, no_response, long_chat]
    ds = InstructionDataset(data, StubTokenizer(), allowed_max_length=256)
    assert len(ds) == 1  # no_response skipped, long_chat filtered out

    token_ids, prompt_len = ds[0]
    instruction, response = format_entry(GLAIVE_ENTRY)
    assert prompt_len == len(instruction)  # char-level stub: one token per char
    assert len(token_ids) == len(instruction) + len(response)


def test_collate_shapes_and_shift():
    batch = [([10, 11, 12, 13, 14], 3), ([20, 21, 22], 2)]
    inputs, targets = custom_collate_fn(batch, pad_token_id=0, ignore_index=-100)

    # batch max length = 5 + 1 (appended EOS), minus 1 for the shift
    assert inputs.shape == targets.shape == (2, 5)
    # unmasked targets are the inputs shifted left by one (response region)
    assert torch.equal(inputs[0, 3:], targets[0, 2:4])


def test_collate_masks_prompt_and_padding():
    batch = [([10, 11, 12, 13, 14], 3), ([20, 21, 22], 2)]
    inputs, targets = custom_collate_fn(batch, pad_token_id=0, ignore_index=-100)

    # prompt tokens (before the response) are masked in targets
    assert (targets[0, :2] == -100).all()
    # response tokens keep their values
    assert targets[0, 2].item() == 13
    # first EOS/pad after the sequence is kept, the rest masked
    row1 = targets[1].tolist()
    assert row1.count(0) == 1
    assert row1[-2:] == [-100, -100]


def test_split_glaive_fractions():
    data = list(range(100))
    train, val, test = split_glaive(data)
    assert len(train) == 85 and len(val) == 10 and len(test) == 5
    assert train + val + test == data
