#!/usr/bin/env python3
"""Training script for ASCII art discrete diffusion model.

Uses the LLaDA/MDLM (1/t)-weighted masked diffusion objective via model.compute_loss().
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from data import ASCIIArtDataset, get_tokenizer
from model import ASCIIDiffusionModel


def train(model_cfg: ModelConfig, train_cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tokenizer & dataset
    tokenizer = get_tokenizer()
    dataset = ASCIIArtDataset(data_dir=train_cfg.data_dir)
    rows, cols = dataset.get_grid_shape()
    print(f"Dataset: {len(dataset)} samples, grid {rows}x{cols}, vocab {tokenizer.vocab_size}")

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        drop_last=True,
    )

    # Model
    model = ASCIIDiffusionModel(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        ff_dim=model_cfg.ff_dim,
        max_rows=model_cfg.max_rows,
        max_cols=model_cfg.max_cols,
        dropout=model_cfg.dropout,
    ).to(device)
    print(f"Model parameters: {model.param_count():,}")

    # Optimizer with cosine LR decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    total_steps = train_cfg.epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Output directory
    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train_log.txt"

    global_step = 0
    best_loss = float("inf")

    print(f"Training for {train_cfg.epochs} epochs ({total_steps} steps)")
    print(f"Logging every {train_cfg.log_every} steps, samples every {train_cfg.sample_every} steps")
    print(f"Checkpoints every {train_cfg.save_every} steps -> {output_dir}/")
    print("-" * 60)

    for epoch in range(train_cfg.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in dataloader:
            token_ids = batch["token_ids"].to(device)  # (B, seq_len)

            # LLaDA objective: (1/t)-weighted CE on masked positions
            loss = model.compute_loss(
                token_ids, mask_id=tokenizer.mask_id, rows=rows, cols=cols
            )

            optimizer.zero_grad()
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Logging
            if global_step % train_cfg.log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                lr = scheduler.get_last_lr()[0]
                msg = (
                    f"[epoch {epoch+1}/{train_cfg.epochs}] "
                    f"step {global_step}/{total_steps} | "
                    f"loss {loss.item():.4f} (avg {avg_loss:.4f}) | "
                    f"lr {lr:.2e}"
                )
                print(msg)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")

            # Generate samples
            if global_step % train_cfg.sample_every == 0:
                print(f"\n--- Sample at step {global_step} ---")
                model.eval()
                samples = model.generate(
                    seq_len=rows * cols,
                    rows=rows, cols=cols,
                    mask_id=tokenizer.mask_id,
                    steps=train_cfg.sample_steps,
                    batch_size=train_cfg.num_samples,
                    device=device,
                )
                for i in range(train_cfg.num_samples):
                    art = tokenizer.flat_to_grid(
                        tokenizer.decode(samples[i].tolist(), skip_special=True),
                        rows, cols,
                    )
                    print(art)
                    print(f"--- end sample {i+1} ---\n")
                    sample_path = output_dir / f"sample_step{global_step}_{i}.txt"
                    sample_path.write_text(art, encoding="utf-8")
                model.train()

            # Checkpointing
            if global_step % train_cfg.save_every == 0:
                ckpt_path = output_dir / f"checkpoint_step{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "model_config": vars(model_cfg),
                    "train_config": vars(train_cfg),
                    "loss": loss.item(),
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1} complete | avg loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = output_dir / "best_model.pt"
            torch.save({
                "step": global_step,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_config": vars(model_cfg),
                "loss": best_loss,
            }, best_path)
            print(f"New best model (loss {best_loss:.4f}) -> {best_path}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train ASCII art diffusion model")

    # Model args
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training args
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)

    # Logging
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="runs")

    args = parser.parse_args()

    model_cfg = ModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )

    train_cfg = TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        log_every=args.log_every,
        sample_every=args.sample_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
    )

    train(model_cfg, train_cfg)


if __name__ == "__main__":
    main()
