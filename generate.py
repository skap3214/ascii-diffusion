#!/usr/bin/env python3
"""Inference/sampling script for ASCII art discrete diffusion model.

Uses model.generate() for iterative unmasking with confidence-based remasking.
"""

import argparse

import torch

from data import get_tokenizer
from model import ASCIIDiffusionModel


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]

    model = ASCIIDiffusionModel(
        vocab_size=cfg["vocab_size"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        ff_dim=cfg["ff_dim"],
        max_rows=cfg["max_rows"],
        max_cols=cfg["max_cols"],
        dropout=cfg.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_condition(path: str, tokenizer, rows: int, cols: int):
    """Load a partial ASCII art file for conditioning.

    Encodes the given lines and returns (condition_ids, condition_mask).
    During generation, conditioned positions are filled first and never remasked.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    seq_len = rows * cols
    condition_ids = torch.full((seq_len,), tokenizer.pad_id, dtype=torch.long)
    condition_mask = torch.zeros(seq_len, dtype=torch.bool)

    for r, line in enumerate(lines):
        if r >= rows:
            break
        line = line.rstrip("\n")
        if not line.strip():
            continue
        line = line.ljust(cols)[:cols]
        ids = tokenizer.encode(line)
        start = r * cols
        for c, tid in enumerate(ids):
            condition_ids[start + c] = tid
            condition_mask[start + c] = True

    return condition_ids, condition_mask


def main():
    parser = argparse.ArgumentParser(description="Generate ASCII art with trained diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--steps", type=int, default=64, help="Number of denoising steps")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--rows", type=int, default=40, help="Grid rows")
    parser.add_argument("--cols", type=int, default=80, help="Grid cols")
    parser.add_argument("--condition", type=str, default="", help="Path to partial ASCII art for conditioning")
    parser.add_argument("--output", type=str, default="", help="Save output to file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = get_tokenizer()
    model = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    seq_len = args.rows * args.cols

    # Optional conditioning: pre-fill known positions before generation
    initial_ids = None
    if args.condition:
        condition_ids, condition_mask = load_condition(
            args.condition, tokenizer, args.rows, args.cols
        )
        # Build initial sequence: conditioned positions filled, rest masked
        initial_ids = torch.full((args.num_samples, seq_len), tokenizer.mask_id, dtype=torch.long)
        initial_ids[:, condition_mask] = condition_ids[condition_mask]
        initial_ids = initial_ids.to(device)
        print(f"Conditioning on {condition_mask.sum().item()} tokens from {args.condition}")

    # Generate using model's built-in iterative unmasking
    if initial_ids is not None:
        # For conditioning, we run the model's generate but pass pre-filled input
        # We need to manually handle this since model.generate() starts from all-mask
        samples = _generate_conditioned(
            model, initial_ids, tokenizer.mask_id,
            args.rows, args.cols, args.steps, args.temperature, device,
        )
    else:
        samples = model.generate(
            seq_len=seq_len,
            rows=args.rows, cols=args.cols,
            mask_id=tokenizer.mask_id,
            steps=args.steps,
            temperature=args.temperature,
            batch_size=args.num_samples,
            device=device,
        )

    results = []
    for i in range(args.num_samples):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{args.num_samples} ({args.steps} steps, temp={args.temperature})")
        print("=" * 60)
        art = tokenizer.flat_to_grid(
            tokenizer.decode(samples[i].tolist(), skip_special=True),
            args.rows, args.cols,
        )
        print(art)
        results.append(art)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for i, art in enumerate(results):
                if i > 0:
                    f.write("\n\n" + "=" * 60 + "\n\n")
                f.write(art)
        print(f"\nSaved to {args.output}")


@torch.no_grad()
def _generate_conditioned(
    model, initial_ids, mask_id, rows, cols, steps, temperature, device,
):
    """Generate with partial conditioning by running iterative unmasking on pre-filled input."""
    import torch.nn.functional as F

    x = initial_ids.clone()
    batch_size, seq_len = x.shape
    # Track which positions are conditioned (never remask these)
    conditioned = x != mask_id

    for step in range(steps):
        t_now = 1.0 - step / steps
        t_next = 1.0 - (step + 1) / steps

        is_masked = (x == mask_id)
        n_masked = is_masked.float().sum(dim=1)

        n_to_unmask = (n_masked * (1.0 - t_next / max(t_now, 1e-8))).long().clamp(min=1)

        t_tensor = torch.full((batch_size,), t_now, device=device)
        logits = model.forward(x, t_tensor, rows=rows, cols=cols)

        probs = F.softmax(logits / temperature, dim=-1)
        sampled = torch.multinomial(probs.view(-1, model.vocab_size), 1).view(batch_size, seq_len)

        confidence = probs.gather(2, sampled.unsqueeze(-1)).squeeze(-1)
        confidence[~is_masked] = float("inf")

        for b in range(batch_size):
            if n_masked[b] == 0:
                continue
            k = min(int(n_to_unmask[b].item()), int(n_masked[b].item()))
            masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
            masked_conf = confidence[b, masked_indices]
            _, topk_rel = masked_conf.topk(k, largest=True)
            unmask_indices = masked_indices[topk_rel]
            x[b, unmask_indices] = sampled[b, unmask_indices]

    # Final cleanup
    is_masked = (x == mask_id)
    if is_masked.any():
        t_tensor = torch.full((batch_size,), 0.01, device=device)
        logits = model.forward(x, t_tensor, rows=rows, cols=cols)
        preds = logits.argmax(dim=-1)
        x[is_masked] = preds[is_masked]

    return x


if __name__ == "__main__":
    main()
