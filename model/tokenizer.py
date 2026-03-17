"""Character-level tokenizer for ASCII art diffusion."""

import torch


class ASCIITokenizer:
    """Tokenizer for printable ASCII characters plus special tokens.

    Vocabulary:
      0: [PAD]
      1: [MASK]
      2..96: printable ASCII chars (space 0x20 through tilde 0x7E)
    """

    PAD_ID = 0
    MASK_ID = 1
    VOCAB_OFFSET = 2

    def __init__(self):
        # 95 printable ASCII chars: space (32) through ~ (126)
        self.printable = [chr(i) for i in range(32, 127)]
        self.char_to_id = {ch: i + self.VOCAB_OFFSET for i, ch in enumerate(self.printable)}
        self.id_to_char = {i + self.VOCAB_OFFSET: ch for i, ch in enumerate(self.printable)}
        self.id_to_char[self.PAD_ID] = ""
        self.id_to_char[self.MASK_ID] = "\u2588"  # block char for visualization
        self.vocab_size = len(self.printable) + 2  # +PAD +MASK = 97

    def encode(self, text: str) -> list[int]:
        """Encode a string to token IDs. Unknown chars become space."""
        return [self.char_to_id.get(ch, self.char_to_id[" "]) for ch in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """Decode token IDs back to a string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.id_to_char.get(i, " ") for i in ids)

    def encode_grid(self, lines: list[str], rows: int, cols: int) -> torch.Tensor:
        """Encode a list of text lines into a flat (rows*cols,) tensor, padded/truncated to fit."""
        token_ids = []
        for r in range(rows):
            line = lines[r] if r < len(lines) else ""
            # Pad or truncate each line to cols
            line = line.ljust(cols)[:cols]
            token_ids.extend(self.encode(line))
        return torch.tensor(token_ids, dtype=torch.long)

    def decode_grid(self, ids: list[int] | torch.Tensor, cols: int) -> str:
        """Decode a flat sequence of IDs into a multi-line string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        text = self.decode(ids)
        lines = [text[i : i + cols] for i in range(0, len(text), cols)]
        return "\n".join(line.rstrip() for line in lines)
