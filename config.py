"""Shared configuration for ASCII art diffusion training and generation."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 100  # 5 special + 95 printable ASCII
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 4
    ff_dim: int = 1024
    max_rows: int = 40
    max_cols: int = 80
    dropout: float = 0.1


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data/processed"
    batch_size: int = 32
    num_workers: int = 0

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 100
    grad_clip: float = 1.0

    # Logging & checkpointing
    log_every: int = 50  # steps
    sample_every: int = 500  # steps
    save_every: int = 1000  # steps
    output_dir: str = "runs"

    # Generation samples during training
    sample_steps: int = 50
    num_samples: int = 2


@dataclass
class GenerateConfig:
    checkpoint: str = ""
    steps: int = 64
    temperature: float = 1.0
    num_samples: int = 1
    rows: int = 40
    cols: int = 80
    # Partial conditioning: path to a text file with initial rows
    condition_file: str = ""
