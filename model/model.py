"""Discrete diffusion model for ASCII art generation.

Based on the masked diffusion approach (LLaDA, Sahoo et al. MDLM):
- Forward process: mask tokens with probability t ~ U[0,1]
- Reverse process: transformer predicts masked tokens
- Loss: (1/t)-weighted cross-entropy on masked positions (likelihood lower bound)
- Inference: iterative unmasking with low-confidence remasking
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for continuous scalar values (e.g., mask rate / timestep)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch,) float in [0, 1] -> (batch, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / half
        )
        args = x[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class ASCIIDiffusionModel(nn.Module):
    """Discrete diffusion model for ASCII art.

    Uses variable-rate masking (0% to 100%) as the noise schedule.
    At each training step, a random fraction of tokens are replaced with [MASK],
    and the model learns to predict the original tokens at all positions.

    Args:
        vocab_size: Number of tokens (100 = 95 printable ASCII + 5 special tokens).
        hidden_dim: Transformer hidden dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward intermediate dimension.
        max_rows: Maximum grid height.
        max_cols: Maximum grid width.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        ff_dim: int = 1024,
        max_rows: int = 40,
        max_cols: int = 80,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_rows = max_rows
        self.max_cols = max_cols

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)

        # 2D positional embeddings
        self.row_emb = nn.Embedding(max_rows, hidden_dim)
        self.col_emb = nn.Embedding(max_cols, hidden_dim)

        # Timestep / mask-rate conditioning
        self.time_emb = SinusoidalEmbedding(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer encoder stack
        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output head
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_2d_positions(self, batch_size: int, rows: int, cols: int, device: torch.device):
        """Generate row and column position tensors for a grid."""
        row_pos = torch.arange(rows, device=device).unsqueeze(1).expand(rows, cols).reshape(-1)
        col_pos = torch.arange(cols, device=device).unsqueeze(0).expand(rows, cols).reshape(-1)
        return row_pos.unsqueeze(0).expand(batch_size, -1), col_pos.unsqueeze(0).expand(batch_size, -1)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask_rate: torch.Tensor,
        row_positions: torch.Tensor | None = None,
        col_positions: torch.Tensor | None = None,
        rows: int | None = None,
        cols: int | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: (batch, seq_len) token indices (may include [MASK] tokens).
            mask_rate: (batch,) float in [0, 1], the fraction of tokens masked.
            row_positions: (batch, seq_len) row index for each token. Auto-generated if None.
            col_positions: (batch, seq_len) column index for each token. Auto-generated if None.
            rows: Grid height, required if row_positions is None.
            cols: Grid width, required if col_positions is None.
            padding_mask: (batch, seq_len) bool, True for positions to ignore.

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, S = token_ids.shape
        device = token_ids.device

        # Auto-generate 2D positions if not provided
        if row_positions is None or col_positions is None:
            assert rows is not None and cols is not None, "Provide rows/cols or explicit positions"
            row_positions, col_positions = self._make_2d_positions(B, rows, cols, device)

        # Embeddings: token + row + col
        x = self.token_emb(token_ids) + self.row_emb(row_positions) + self.col_emb(col_positions)

        # Add timestep conditioning (broadcast across sequence)
        t = self.time_proj(self.time_emb(mask_rate))  # (B, hidden_dim)
        x = x + t.unsqueeze(1)

        # Transformer encoder
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.final_norm(x)
        logits = self.output_proj(x)  # (B, S, vocab_size)
        return logits

    def compute_loss(
        self,
        token_ids: torch.Tensor,
        mask_id: int,
        rows: int = 40,
        cols: int = 80,
    ) -> torch.Tensor:
        """Full training step: sample mask rate, mask tokens, compute 1/t-weighted loss.

        Implements the LLaDA/MDLM objective:
            L = E_{t~U[0,1]} [ (1/t) * sum_{masked} -log p(x_i | x_t) ]

        Args:
            token_ids: (batch, seq_len) ground truth token IDs.
            mask_id: Token ID for [MASK].
            rows: Grid height.
            cols: Grid width.

        Returns:
            Scalar loss (mean over batch).
        """
        B, S = token_ids.shape
        device = token_ids.device

        # Sample t ~ U(eps, 1) per example (avoid t=0 which gives 1/t=inf)
        t = torch.rand(B, device=device) * 0.99 + 0.01  # [0.01, 1.0]

        # Create mask: each token masked independently with probability t
        mask_probs = t[:, None].expand(B, S)  # (B, S)
        mask = torch.bernoulli(mask_probs).bool()

        # Apply mask
        masked_ids = token_ids.clone()
        masked_ids[mask] = mask_id

        # Forward
        logits = self.forward(masked_ids, t, rows=rows, cols=cols)

        # Cross-entropy only on masked positions, weighted by 1/t
        loss_per_token = F.cross_entropy(
            logits.view(-1, self.vocab_size), token_ids.view(-1), reduction="none"
        ).view(B, S)

        # Zero out loss on unmasked positions
        loss_per_token = loss_per_token * mask.float()

        # Per-example: (1/t) * mean over masked tokens
        num_masked = mask.float().sum(dim=1).clamp(min=1)
        loss_per_example = (1.0 / t) * (loss_per_token.sum(dim=1) / num_masked)

        return loss_per_example.mean()

    @torch.no_grad()
    def generate(
        self,
        seq_len: int = 3200,
        rows: int = 40,
        cols: int = 80,
        mask_id: int = 3,
        steps: int = 50,
        temperature: float = 1.0,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Generate ASCII art via iterative unmasking with low-confidence remasking.

        Starts from fully masked sequence, progressively unmasks tokens from
        high to low mask rate. At each step, predicts all masked positions,
        keeps the most confident predictions, and remasks the rest.

        Args:
            seq_len: Total sequence length (rows * cols).
            rows: Grid height.
            cols: Grid width.
            mask_id: Token ID for [MASK].
            steps: Number of denoising steps.
            temperature: Sampling temperature.
            batch_size: Number of samples to generate.
            device: Device to generate on.

        Returns:
            token_ids: (batch_size, seq_len) generated token IDs.
        """
        # Start fully masked
        x = torch.full((batch_size, seq_len), mask_id, dtype=torch.long, device=device)

        for step in range(steps):
            # Current and next mask rate
            t_now = 1.0 - step / steps
            t_next = 1.0 - (step + 1) / steps

            # Which positions are still masked
            is_masked = (x == mask_id)
            n_masked = is_masked.float().sum(dim=1)  # (B,)

            # How many to unmask this step
            n_to_unmask = (n_masked * (1.0 - t_next / max(t_now, 1e-8))).long().clamp(min=1)

            # Get predictions
            t_tensor = torch.full((batch_size,), t_now, device=device)
            logits = self.forward(x, t_tensor, rows=rows, cols=cols)

            # Sample from logits
            probs = F.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, seq_len)

            # Confidence = probability of the sampled token
            confidence = probs.gather(2, sampled.unsqueeze(-1)).squeeze(-1)  # (B, S)
            # Only consider masked positions
            confidence[~is_masked] = float("inf")

            # For each example, unmask the top-k most confident masked positions
            for b in range(batch_size):
                if n_masked[b] == 0:
                    continue
                k = min(int(n_to_unmask[b].item()), int(n_masked[b].item()))
                masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
                masked_conf = confidence[b, masked_indices]
                _, topk_rel = masked_conf.topk(k, largest=True)
                unmask_indices = masked_indices[topk_rel]
                x[b, unmask_indices] = sampled[b, unmask_indices]

        # Final pass: fill any remaining masks
        is_masked = (x == mask_id)
        if is_masked.any():
            t_tensor = torch.full((batch_size,), 0.01, device=device)
            logits = self.forward(x, t_tensor, rows=rows, cols=cols)
            probs = F.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, seq_len)
            x[is_masked] = sampled[is_masked]

        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
