import os
import requests
import json
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch.nn as nn


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()



def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def format_entry_old(entry):
    system = entry['system'].strip().replace('SYSTEM', '###SYSTEM')

    turns = [t.strip().replace("<|endoftext|>", "").replace('USER', '###USER').replace('ASSISTANT', '###ASSISTANT') for t in entry['chat'].split('\n\n\n') if t.strip()]

    instruction = ""
    response = ""

    for turn in turns:
        if (turn.startswith('###ASSISTANT:') and ('<functioncall>' in turn or 'sorry' in turn)):
            response = turn
            break
        else:
            instruction = instruction + '\n' +  turn


    return system + '\n' + instruction, response


def format_entry(entry):
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
    allowed_max_length = 1024

    def __init__(self, data, tokenizer):
        self.data = data

        self.encoded_texts = []

        for entry in data:            
            prompt, response = format_entry(entry)                                                                                                                                           
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
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    batch_max_length = max(len(token_ids)+1 for token_ids, _ in batch)
    if allowed_max_length is not None:
        batch_max_length = min(batch_max_length, allowed_max_length)
                                                                                                                                                                                                                
    inputs_lst, targets_lst = [], []
                                                                                                                                                                                                                
    for token_ids, prompt_len in batch:
        new_item = token_ids.copy()
        new_item += [pad_token_id]  # token for end of sequence
        if allowed_max_length is not None:
            new_item = new_item[:allowed_max_length]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))  # rest are paddings
                                                                                                                                                                                                                
        inputs = torch.tensor(padded[:-1])                                                                                                                                                                      
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id  # get where it is 50256 and have them True. 
                                        # Ex [1,2,50256, 50256] ->[False, False, True, True]
        
        indices = torch.nonzero(mask).squeeze() # Get their indices : [2, 3]
        
        if indices.numel() > 1:                 #Check if indices have elements more than 1. 
                                                #If it is 1 or less, ignore, as it is for eos token
            targets[indices[1:]] = ignore_index # Set -100 for all pads, except for eos token. note: eos and pads use 50256 idx                                                                                                                                                           

        targets[:prompt_len - 1] = ignore_index  # mask all prompt tokens from the targets. prompt_len - 1 because of targets = torch.tensor(padded[1:])
                                                                                                         
                
        inputs_lst.append(inputs)
        targets_lst.append(targets)                                                                                                                                                                             
                
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor                                                                                                                                                                        


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, num_heads: int, context_length: int, qkv_bias: bool, dropout: float) -> None:
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out          = d_out
        self.num_heads      = num_heads
        self.head_dim       = d_out // num_heads

        self.W_query        = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key          = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value        = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj       = nn.Linear(d_out, d_out)

        self.dropout        = nn.Dropout(dropout)

        # Pre-allocate causal mask at max size; sliced to actual length in forward
        self.register_buffer('mask', torch.triu(torch.ones((context_length, context_length)), diagonal=1))


    def forward(self, x: torch.Tensor):
        b, num_tokens, _ = x.shape

        # Project input into query, key, value spaces
        # [b, num_tokens, d_out]
        queries     = self.W_query(x)
        keys        = self.W_key(x)
        values      = self.W_value(x)

        # Split d_out into separate heads: each head gets its own head_dim slice of the embedding
        # We can't view directly to (b, num_heads, num_tokens, head_dim)
        # because view doesn't rearrange memory — it would mix tokens and heads
        # [b, num_tokens, d_out] -> [b, num_tokens, num_heads, head_dim]
        queries     = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys        = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values      = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Move num_heads before num_tokens so each head can do its own attention independently
        # [b, num_tokens, num_heads, head_dim] -> [b, num_heads, num_tokens, head_dim]
        queries     = queries.transpose(1, 2)
        keys        = keys.transpose(1, 2)
        values      = values.transpose(1, 2)

        # Dot product between queries and keys for each head
        # [b, num_heads, num_tokens, head_dim] @ [b, num_heads, head_dim, num_tokens]
        # -> [b, num_heads, num_tokens, num_tokens]
        attn_scores = queries @ keys.transpose(2, 3)

        # Upper triangular mask: prevent token i from attending to future tokens j > i
        # Mask is pre-allocated at max context_length, sliced to actual sequence length
        # [num_tokens, num_tokens]
        masked_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(masked_bool, -torch.inf)

        # Scale by sqrt(head_dim) to prevent dot products from growing too large,
        # which would push softmax into regions with tiny gradients
        # Softmax over last dim (key positions) so weights sum to 1 for each query
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)

        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values using attention weights
        # [b, num_heads, num_tokens, num_tokens] @ [b, num_heads, num_tokens, head_dim]
        # -> [b, num_heads, num_tokens, head_dim]
        # Transpose to group heads per token for concatenation
        # -> [b, num_tokens, num_heads, head_dim]
        context_vectors = (attn_weights @ values).transpose(1, 2)

        # Concatenate all heads back: (num_heads * head_dim) = d_out
        # [b, num_tokens, d_out]
        context_vectors = context_vectors.reshape(b, num_tokens, self.d_out)

        # Final linear projection to mix information across heads
        # [b, num_tokens, d_out]
        context_vectors = self.out_proj(context_vectors)

        return context_vectors


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
    

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]), 
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            context_length=cfg["context_length"],
            qkv_bias=cfg["qkv_bias"],
            dropout=cfg["drop_rate"],
        )
        self.layer_norm1 = LayerNorm(cfg["emb_dim"])
        self.layer_norm2 = LayerNorm(cfg["emb_dim"])
        self.feed_forward = FeedForward(cfg)
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.drop_resid(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm2(x)
        x = self.attention(x)
        x = self.drop_resid(x)
        x = x + shortcut
        
        return x
    

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond  = idx[:,-context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        
        next_idx = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.concat((idx, next_idx), dim=1)

    return idx


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, k=top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
        
        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values

            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)

        else:
            next_idx = torch.argmax(logits, dim=-1, keepdim=True)

        if next_idx == eos_id:
            break

        idx = torch.cat((idx, next_idx), dim=1)

    return idx


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

