# ASCII Diffusion — Handoff Summary

## What this project is

A discrete text diffusion model that generates ASCII art. Instead of generating text left-to-right like GPT, it starts from a grid of [MASK] tokens and iteratively unmasks them — like image diffusion but for characters.

## Approach

Based on two key references:
- **RoBERTa Diffusion blog post** (https://nathan.rs/posts/roberta-diffusion/) — showed that BERT's MLM objective is equivalent to one step of discrete text diffusion. Train with variable masking rates → get a diffusion model.
- **LLaDA paper** (https://arxiv.org/abs/2502.09992, https://github.com/ML-GSAI/LLaDA) — scaled masked diffusion to 8B params, matching LLaMA3 8B. We use their (1/t)-weighted ELBO loss formulation.

We chose discrete masking diffusion over continuous embedding diffusion because ASCII art is 2D spatial (not sequential), has a tiny vocabulary (~95 chars), and diffusion's parallel denoising naturally fits spatial generation.

## What's been built

### Project structure (`/ascii-diffusion/`)
- `config.py` — Dataclasses for model, training, and generation config
- `model/model.py` — Transformer encoder with 2D positional embeddings (row + col), masking rate conditioning, LLaDA-based `compute_loss()` and `generate()` methods
- `model/tokenizer.py` — Character-level tokenizer (~100 tokens: 95 printable ASCII + MASK + PAD + specials)
- `data/dataset.py` — PyTorch Dataset class
- `data/tokenizer.py` — Data-side tokenizer utilities
- `data/processed/` — Processed dataset (4,566 ASCII art pieces as JSONL, vocab=100, seq_len=3200 = 40 rows x 80 cols)
- `raw_data/` — Raw ASCII art files (~7,000 files from various sources)
- `scripts/` — Data collection/processing scripts
- `train.py` — Training script with LLaDA ELBO objective, cosine LR decay, gradient clipping, logging, sample generation, checkpointing
- `generate.py` — Inference script with confidence-based unmasking and optional partial conditioning
- `requirements.txt` — torch>=2.0.0, numpy>=1.24.0

### Model specs
- ~5M parameters
- 6 layers, 256 hidden dim, 4 heads, 1024 FF dim
- 2D positional embeddings (row + column, not just flat sequence position)
- Vocab: ~100 tokens (character-level)
- Sequence length: 3,200 (40 rows x 80 cols)

### GitHub repo
https://github.com/skap3214/ascii-diffusion

## Training so far

Ran on CPU (MacBook, 36GB RAM) with batch_size=4. Completed ~1,100 steps of epoch 1 before stopping.

Loss progression:
- Step 25: avg 5.37
- Step 200: avg 4.63
- Step 500: avg 3.79
- Step 1100: avg 3.18

Loss is trending down. Occasional spikes (up to 40.0) caused by the (1/t) loss weighting when masking rate is very low — this is expected behavior. Sample generations at step 500 and 1000 were mostly blank (too early).

Training was stopped because CPU is too slow (~150 steps/hr, would take ~16 days for 50 epochs).

## What to do next

### Immediate: Train on GPU
1. Clone the repo on a GPU VM (recommended: A40 with 48GB VRAM)
2. `pip install -r requirements.txt`
3. `python3 train.py --epochs 50 --batch-size 32` (or 64 if VRAM allows)
4. With a GPU, batch_size=32 should work. Training should finish in under an hour.
5. Monitor loss — should drop well below 3.0. Check sample generations in `runs/` directory.

### After training: Generate
```bash
python3 generate.py --checkpoint runs/best_model.pt --steps 64 --temperature 1.0
```

### Potential improvements to explore
- **Clamp masking rate** away from 0 (e.g., min 0.05) to reduce loss spikes from (1/t) weighting
- **Gradient accumulation** to simulate larger batch sizes
- **Conditional generation** — add class labels (e.g., "cat", "house") so you can control what gets generated. LLaDA's SFT approach: only mask the art tokens, not the label.
- **Smaller grid** (e.g., 20x40) for faster iteration during experimentation
- **Confidence-based unmasking** refinement — tune how aggressively to unmask at each step
- **Dataset expansion** — more ASCII art, or curated subsets by category
