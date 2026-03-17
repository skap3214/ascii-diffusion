"""
PyTorch Dataset for ASCII art training data.

Loads processed ASCII art from JSONL files and provides tokenized
fixed-size grids for model training.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from .tokenizer import ASCIITokenizer, get_tokenizer

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"


class ASCIIArtDataset(Dataset):
    """
    PyTorch Dataset that loads processed ASCII art pieces.

    Each item is a dict with:
        - "token_ids": LongTensor of shape (seq_len,) — tokenized flat grid
        - "source": str — source identifier for the piece
    """

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
        tokenizer: Optional[ASCIITokenizer] = None,
        add_bos: bool = False,
        add_eos: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else PROCESSED_DIR
        self.tokenizer = tokenizer or get_tokenizer()
        self.add_bos = add_bos
        self.add_eos = add_eos

        # Load metadata
        meta_path = self.data_dir / "metadata.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text())
        else:
            self.metadata = {"grid_rows": 40, "grid_cols": 80}

        self.grid_rows = self.metadata.get("grid_rows", 40)
        self.grid_cols = self.metadata.get("grid_cols", 80)
        self.seq_len = self.grid_rows * self.grid_cols

        # Load data
        self.samples = []
        jsonl_path = self.data_dir / "ascii_art.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
                        if max_samples and len(self.samples) >= max_samples:
                            break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        flat_text = sample["data"]

        token_ids = self.tokenizer.encode(
            flat_text, add_bos=self.add_bos, add_eos=self.add_eos
        )

        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "source": sample.get("source", ""),
        }

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_grid_shape(self) -> tuple[int, int]:
        return (self.grid_rows, self.grid_cols)

    def decode_sample(self, idx: int) -> str:
        """Decode a sample back to a displayable ASCII art grid."""
        sample = self.samples[idx]
        return self.tokenizer.flat_to_grid(
            sample["data"], self.grid_rows, self.grid_cols
        )

    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to displayable ASCII art."""
        ids = token_ids.tolist()
        flat = self.tokenizer.decode(ids, skip_special=True)
        return self.tokenizer.flat_to_grid(flat, self.grid_rows, self.grid_cols)


def create_dataloader(
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """Convenience function to create a DataLoader for ASCII art."""
    dataset = ASCIIArtDataset(**dataset_kwargs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
