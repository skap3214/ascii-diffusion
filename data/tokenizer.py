"""
Character-level tokenizer for ASCII art.

Maps the ~95 printable ASCII characters (codes 32-126) plus special tokens
to integer token IDs for model input/output.
"""

import json
from pathlib import Path


# Special tokens
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
MASK_TOKEN = "<MASK>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN, UNK_TOKEN]

# Printable ASCII: space (32) through tilde (126) = 95 characters
PRINTABLE_ASCII = [chr(c) for c in range(32, 127)]


class ASCIITokenizer:
    """Character-level tokenizer for ASCII art."""

    def __init__(self):
        self.special_tokens = list(SPECIAL_TOKENS)
        self.chars = list(PRINTABLE_ASCII)

        # Build vocab: special tokens first, then printable ASCII
        self.vocab = self.special_tokens + self.chars
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}

        # Convenience IDs
        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]
        self.mask_id = self.token_to_id[MASK_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    @property
    def num_chars(self) -> int:
        return len(self.chars)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode a string to a list of token IDs."""
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in text:
            ids.append(self.token_to_id.get(ch, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode a list of token IDs back to a string."""
        chars = []
        special_ids = set(range(len(self.special_tokens)))
        for token_id in ids:
            if skip_special and token_id in special_ids:
                continue
            token = self.id_to_token.get(token_id, UNK_TOKEN)
            if token in SPECIAL_TOKENS:
                if not skip_special:
                    chars.append(token)
            else:
                chars.append(token)
        return "".join(chars)

    def flat_to_grid(self, flat: str, rows: int = 40, cols: int = 80) -> str:
        """Convert a flat string back to a multi-line grid."""
        lines = []
        for r in range(rows):
            start = r * cols
            end = start + cols
            lines.append(flat[start:end].rstrip())
        # Remove trailing empty lines
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)

    def save(self, path: str | Path):
        """Save tokenizer config to JSON."""
        path = Path(path)
        config = {
            "special_tokens": self.special_tokens,
            "chars": self.chars,
            "vocab_size": self.vocab_size,
        }
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ASCIITokenizer":
        """Load tokenizer from JSON config (validates consistency)."""
        path = Path(path)
        config = json.loads(path.read_text(encoding="utf-8"))
        tokenizer = cls()
        assert tokenizer.vocab_size == config["vocab_size"], \
            f"Vocab size mismatch: {tokenizer.vocab_size} vs {config['vocab_size']}"
        return tokenizer

    def __repr__(self) -> str:
        return f"ASCIITokenizer(vocab_size={self.vocab_size}, special={self.num_special_tokens}, chars={self.num_chars})"


# Module-level singleton for convenience
_default_tokenizer = None


def get_tokenizer() -> ASCIITokenizer:
    """Get or create the default tokenizer instance."""
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = ASCIITokenizer()
    return _default_tokenizer
