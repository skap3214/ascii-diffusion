#!/usr/bin/env python3
"""
Process raw ASCII art files into a normalized, fixed-size grid format
suitable for training.

Each piece is normalized to GRID_ROWS x GRID_COLS, padded with spaces,
and saved as a flat string (row-by-row concatenation).
"""

import json
import re
import sys
from pathlib import Path

# Fixed grid dimensions
GRID_ROWS = 40
GRID_COLS = 80

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "raw_data"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Only keep printable ASCII (codes 32-126) plus newline for processing
PRINTABLE_ASCII = set(chr(c) for c in range(32, 127))


def clean_art(text: str) -> str:
    """Remove non-printable characters, normalize line endings."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace tabs with spaces
    text = text.replace("\t", "    ")
    # Filter to printable ASCII only (keep newlines for splitting)
    cleaned = []
    for ch in text:
        if ch == "\n" or ch in PRINTABLE_ASCII:
            cleaned.append(ch)
        else:
            cleaned.append(" ")  # replace non-printable with space
    return "".join(cleaned)


def normalize_to_grid(text: str, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> str | None:
    """
    Normalize ASCII art to a fixed grid size.
    Returns a flat string of length rows*cols, or None if the art is too small.
    """
    lines = text.split("\n")

    # Remove leading/trailing empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if len(lines) < 3:  # too small
        return None

    # Check if art is too large - if so, crop from center
    art_height = len(lines)
    art_width = max(len(line) for line in lines) if lines else 0

    if art_width < 3:  # too narrow
        return None

    # If art is larger than grid, crop it (centered)
    if art_height > rows:
        start = (art_height - rows) // 2
        lines = lines[start:start + rows]
    if art_width > cols:
        start = (art_width - cols) // 2
        lines = [line[start:start + cols] for line in lines]

    # Center the art vertically
    art_height = len(lines)
    top_padding = (rows - art_height) // 2

    # Build the grid
    grid_lines = []
    for r in range(rows):
        art_row = r - top_padding
        if 0 <= art_row < len(lines):
            line = lines[art_row]
            # Pad or truncate to cols width, center horizontally
            if len(line) < cols:
                left_pad = (cols - len(line)) // 2
                line = " " * left_pad + line
            line = line[:cols].ljust(cols)
        else:
            line = " " * cols
        grid_lines.append(line)

    flat = "".join(grid_lines)
    assert len(flat) == rows * cols, f"Expected {rows * cols}, got {len(flat)}"

    # Skip if it's mostly empty (less than 2% non-space chars)
    non_space = sum(1 for c in flat if c != " ")
    if non_space < (rows * cols * 0.02):
        return None

    return flat


def process_all():
    """Process all raw ASCII art files into normalized grids."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = list(RAW_DATA_DIR.glob("*.txt"))
    raw_files = [f for f in raw_files if f.name != "manifest.txt"]

    print(f"Found {len(raw_files)} raw files")

    processed = []
    skipped = 0

    for filepath in raw_files:
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  Error reading {filepath.name}: {e}")
            skipped += 1
            continue

        text = clean_art(text)
        flat = normalize_to_grid(text)
        if flat is None:
            skipped += 1
            continue

        processed.append({
            "source": filepath.stem,
            "data": flat,
        })

    print(f"Processed: {len(processed)}, Skipped: {skipped}")

    # Save as JSON lines format
    output_path = PROCESSED_DIR / "ascii_art.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")

    # Also save metadata
    meta = {
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "total_pieces": len(processed),
        "flat_length": GRID_ROWS * GRID_COLS,
    }
    meta_path = PROCESSED_DIR / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Output saved to: {output_path}")
    print(f"Metadata saved to: {meta_path}")
    print(f"Grid size: {GRID_ROWS} rows x {GRID_COLS} cols = {GRID_ROWS * GRID_COLS} chars per piece")

    return len(processed)


if __name__ == "__main__":
    count = process_all()
    if count < 100:
        print(f"\nWARNING: Only {count} pieces processed. You may want to collect more data.")
    sys.exit(0)
