import pandas as pd
import torch
from torch.utils.data import Dataset

from gpt2fc.config import EOS_TOKEN_ID

GLAIVE_URL = "hf://datasets/glaiveai/glaive-function-calling-v2/glaive-function-calling-v2.json"

TRAIN_FRAC = 0.85
VAL_FRAC = 0.10


def load_glaive(data_slice=None):
    """Load Glaive Function Calling v2 as a list of {'system', 'chat'} records."""
    df = pd.read_json(GLAIVE_URL)
    data = df.to_dict("records")
    if data_slice is not None and data_slice > 0:
        data = data[:data_slice]
    return data


def split_glaive(data):
    """Deterministic 85/10/5 train/val/test split."""
    train_end = int(len(data) * TRAIN_FRAC)
    val_end = train_end + int(len(data) * VAL_FRAC)
    return data[:train_end], data[train_end:val_end], data[val_end:]


def format_entry(entry):
    """Convert a Glaive record into an (instruction, response) pair.

    The instruction is the function schema plus all turns before the first
    assistant turn; the response is the first `###ASSISTANT:` turn. Returns
    response="" when the chat contains no assistant turn.
    """
    system = entry['system'].strip().replace('SYSTEM:', '###SYSTEM:')

    turns = [
        t.strip()
         .replace("<|endoftext|>", "")
         .replace('USER:', '###USER:')
         .replace('ASSISTANT:', '###ASSISTANT:')
        for t in entry['chat'].split('\n\n\n')
        if t.strip()
    ]

    instruction_turns = []
    response = ""

    for turn in turns:
        if turn.startswith('###ASSISTANT:'):
            response = turn
            break
        instruction_turns.append(turn)

    instruction = system + '\n' + '\n'.join(instruction_turns)
    return instruction, response


class InstructionDataset(Dataset):
    """Pre-tokenizes (instruction + response) pairs, keeping the prompt length
    so the collate function can mask prompt tokens out of the loss."""

    def __init__(self, data, tokenizer, allowed_max_length=1024):
        self.encoded_texts = []
        self.allowed_max_length = allowed_max_length

        for entry in data:
            prompt, response = format_entry(entry)
            if not response:
                continue

            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response)
            token_ids = prompt_ids + response_ids

            if len(token_ids) + 1 > self.allowed_max_length:
                continue

            self.encoded_texts.append((token_ids, len(prompt_ids)))

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, index):
        return self.encoded_texts[index]


def custom_collate_fn(
    batch,
    pad_token_id=EOS_TOKEN_ID,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """Pad to the batch max length, shift targets by +1, and mask with ignore_index:
    padding tokens beyond the first EOS, and all prompt tokens (loss on response only)."""
    batch_max_length = max(len(token_ids) + 1 for token_ids, _ in batch)
    if allowed_max_length is not None:
        batch_max_length = min(batch_max_length, allowed_max_length)

    inputs_lst, targets_lst = [], []

    for token_ids, prompt_len in batch:
        new_item = token_ids.copy()
        new_item += [pad_token_id]  # EOS token
        if allowed_max_length is not None:
            new_item = new_item[:allowed_max_length]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        targets[:prompt_len - 1] = ignore_index

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
