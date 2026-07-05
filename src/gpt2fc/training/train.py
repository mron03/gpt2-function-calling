import argparse
import functools
import os
import time

import matplotlib
import torch
from torch.utils.data import DataLoader

from gpt2fc.config import EOS_TOKEN_ID, MODEL_CONFIGS, get_device, get_model_config
from gpt2fc.inference.generate import get_tokenizer
from gpt2fc.model import GPTModel
from gpt2fc.model.pretrained import download_and_load_gpt2, load_weights_into_gpt
from gpt2fc.training.data import InstructionDataset, custom_collate_fn, format_entry, load_glaive, split_glaive

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    num_batches = min(num_batches, len(data_loader)) if num_batches else len(data_loader)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.context_length
    encoded = torch.tensor(
        tokenizer.encode(start_context, allowed_special={"<|endoftext|>"})
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        idx = encoded
        for _ in range(50):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            if idx_next == EOS_TOKEN_ID:
                break
            idx = torch.cat((idx, idx_next), dim=1)
        decoded = tokenizer.decode(idx.squeeze(0).tolist())
        print(decoded.replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, output_path):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Loss plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on Glaive function calling")
    parser.add_argument("--model-size",         default="124M",    choices=list(MODEL_CONFIGS))
    parser.add_argument("--num-epochs",         type=int,   default=2)
    parser.add_argument("--batch-size",         type=int,   default=2)
    parser.add_argument("--allowed-max-length", type=int,   default=512)
    parser.add_argument("--lr",                 type=float, default=5e-5)
    parser.add_argument("--weight-decay",       type=float, default=0.1)
    parser.add_argument("--data-slice",         type=int,   default=1000,
                        help="Number of dataset samples to use (use -1 for full dataset)")
    parser.add_argument("--eval-freq",          type=int,   default=5)
    parser.add_argument("--eval-iter",          type=int,   default=5)
    parser.add_argument("--models-dir",         default="weights/gpt2",
                        help="Directory to store/load the pretrained GPT-2 TF checkpoints")
    parser.add_argument("--output-dir",         default="checkpoints")
    parser.add_argument("--device",             default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="Note: PyTorch MPS backward kernels can produce NaN gradients; use cpu/cuda for training")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device(args.device)
    print(f"Device: {device}")

    print("Loading Glaive function calling dataset...")
    data = load_glaive(args.data_slice)
    print(f"Loaded {len(data)} samples")

    train_data, val_data, test_data = split_glaive(data)
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    tokenizer = get_tokenizer()
    collate_fn = functools.partial(custom_collate_fn, allowed_max_length=args.allowed_max_length)

    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer, allowed_max_length=args.allowed_max_length),
        batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        InstructionDataset(val_data, tokenizer, allowed_max_length=args.allowed_max_length),
        batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, drop_last=False,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    cfg = get_model_config(args.model_size)
    print(f"Downloading/loading GPT-2 {args.model_size} weights...")
    _, params = download_and_load_gpt2(model_size=args.model_size, models_dir=args.models_dir)
    model = GPTModel(cfg)
    load_weights_into_gpt(model, params)
    model.to(device)
    print(f"GPT-2 {args.model_size} loaded and weights transferred")

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    print(f"Baseline — Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_context = format_entry(val_data[0])[0]

    start_time = time.time()
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        start_context=start_context,
        tokenizer=tokenizer,
    )
    elapsed = (time.time() - start_time) / 60
    print(f"Training completed in {elapsed:.2f} minutes")

    ckpt_path = os.path.join(args.output_dir, f"gpt2-{args.model_size}-function-calling.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    if train_losses:
        epochs_seen = torch.linspace(0, args.num_epochs, len(train_losses))
        plot_losses(epochs_seen, tokens_seen, train_losses, val_losses,
                    output_path=os.path.join(args.output_dir, "loss-plot.png"))


if __name__ == "__main__":
    main()
