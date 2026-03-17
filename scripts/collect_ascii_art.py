#!/usr/bin/env python3
"""
Collect ASCII art from multiple sources and save as individual text files.

Sources:
1. asweigart/asciiartjsondb (GitHub) - JSON database scraped from asciiart.eu
2. Procedurally generated simple ASCII art patterns (fallback/augmentation)
3. Christopher Johnson's ASCII Art Collection via asciiart.website
"""

import json
import os
import re
import sys
import time
import random
import hashlib
import urllib.request
import urllib.error
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "raw_data"
ASCIIART_JSON_URL = "https://raw.githubusercontent.com/asweigart/asciiartjsondb/main/asciiartdb-asciiarteu.json"

# Categories to scrape from asciiart.website
ASCIIART_WEBSITE_CATEGORIES = [
    "animals", "art", "computers", "food", "holidays", "movies",
    "nature", "people", "plants", "religion", "science", "space",
    "sports", "television", "vehicles",
]


def download_json_db(output_dir: Path) -> list[str]:
    """Download ASCII art from the asweigart/asciiartjsondb GitHub repo."""
    print("Downloading asciiartjsondb from GitHub...")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        req = urllib.request.Request(ASCIIART_JSON_URL, headers={"User-Agent": "ascii-diffusion-collector/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"  Failed to download JSON DB: {e}")
        return []

    arts = []
    if isinstance(data, dict):
        # The JSON is typically {category: {name: art_string, ...}, ...}
        for category, entries in data.items():
            if isinstance(entries, dict):
                for name, art in entries.items():
                    if isinstance(art, str) and len(art.strip()) > 10:
                        arts.append(art.strip())
            elif isinstance(entries, list):
                for art in entries:
                    if isinstance(art, str) and len(art.strip()) > 10:
                        arts.append(art.strip())
            elif isinstance(entries, str) and len(entries.strip()) > 10:
                arts.append(entries.strip())
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str) and len(item.strip()) > 10:
                arts.append(item.strip())
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str) and len(v.strip()) > 10:
                        arts.append(v.strip())

    saved = []
    for i, art in enumerate(arts):
        art_hash = hashlib.md5(art.encode()).hexdigest()[:12]
        filename = f"jsondb_{art_hash}.txt"
        filepath = output_dir / filename
        filepath.write_text(art, encoding="utf-8")
        saved.append(str(filepath))

    print(f"  Saved {len(saved)} pieces from asciiartjsondb")
    return saved


def scrape_asciiart_website(output_dir: Path) -> list[str]:
    """Scrape ASCII art from asciiart.website (Christopher Johnson's collection)."""
    print("Scraping asciiart.website...")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    base_url = "https://asciiart.website/index.php?art={category}"

    for category in ASCIIART_WEBSITE_CATEGORIES:
        url = base_url.format(category=category)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ascii-diffusion-collector/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as e:
            print(f"  Failed to fetch {category}: {e}")
            continue

        # Extract <pre> blocks which typically contain ASCII art
        pre_blocks = re.findall(r"<pre[^>]*>(.*?)</pre>", html, re.DOTALL | re.IGNORECASE)
        for block in pre_blocks:
            # Clean HTML entities
            art = block.replace("&lt;", "<").replace("&gt;", ">")
            art = art.replace("&amp;", "&").replace("&quot;", '"')
            art = re.sub(r"<[^>]+>", "", art)  # strip remaining tags
            art = art.strip()
            if len(art) > 20 and "\n" in art:  # must be multi-line
                art_hash = hashlib.md5(art.encode()).hexdigest()[:12]
                filename = f"website_{category}_{art_hash}.txt"
                filepath = output_dir / filename
                filepath.write_text(art, encoding="utf-8")
                saved.append(str(filepath))

        time.sleep(0.5)  # be polite

    print(f"  Saved {len(saved)} pieces from asciiart.website")
    return saved


def generate_procedural_art(output_dir: Path, count: int = 500) -> list[str]:
    """Generate simple procedural ASCII art patterns as augmentation data."""
    print(f"Generating {count} procedural ASCII art pieces...")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    random.seed(42)

    charset_sets = [
        list("/-\\|*+.oO#@"),
        list("█▓▒░ "),
        list(".:;+=xX#@"),
        list("/\\|-_~*+"),
        list(".oO0@#"),
    ]

    def make_diamond(size: int, char: str = "*", fill: str = " ") -> str:
        lines = []
        for i in range(size):
            spaces = abs(size // 2 - i)
            width = size - 2 * spaces
            if width <= 0:
                continue
            line = fill * spaces + char * width + fill * spaces
            lines.append(line)
        return "\n".join(lines)

    def make_box(w: int, h: int, border: str = "#", fill: str = " ") -> str:
        lines = [border * w]
        for _ in range(h - 2):
            lines.append(border + fill * (w - 2) + border)
        lines.append(border * w)
        return "\n".join(lines)

    def make_triangle(size: int, char: str = "*") -> str:
        lines = []
        for i in range(1, size + 1):
            spaces = " " * (size - i)
            stars = char * (2 * i - 1)
            lines.append(spaces + stars)
        return "\n".join(lines)

    def make_spiral(size: int, chars: list[str]) -> str:
        grid = [[" "] * size for _ in range(size)]
        x, y = size // 2, size // 2
        dx, dy = 1, 0
        steps = 1
        placed = 0
        ci = 0
        while 0 <= x < size and 0 <= y < size and placed < size * size:
            for _ in range(steps):
                if 0 <= x < size and 0 <= y < size:
                    grid[y][x] = chars[ci % len(chars)]
                    ci += 1
                    placed += 1
                x += dx
                y += dy
            dx, dy = -dy, dx
            if dy == 0:
                steps += 1
        return "\n".join("".join(row) for row in grid)

    def make_wave(w: int, h: int, chars: list[str]) -> str:
        import math
        lines = []
        for y in range(h):
            line = []
            for x in range(w):
                val = math.sin(x * 0.3 + y * 0.5) * 0.5 + 0.5
                idx = int(val * (len(chars) - 1))
                line.append(chars[idx])
            lines.append("".join(line))
        return "\n".join(lines)

    def make_random_pattern(w: int, h: int, chars: list[str], density: float = 0.3) -> str:
        lines = []
        for _ in range(h):
            line = []
            for _ in range(w):
                if random.random() < density:
                    line.append(random.choice(chars))
                else:
                    line.append(" ")
            lines.append("".join(line))
        return "\n".join(lines)

    def make_checkerboard(w: int, h: int, c1: str = "#", c2: str = " ") -> str:
        lines = []
        for y in range(h):
            line = []
            for x in range(w):
                line.append(c1 if (x + y) % 2 == 0 else c2)
            lines.append("".join(line))
        return "\n".join(lines)

    generators = [
        lambda: make_diamond(random.randint(7, 25), random.choice("*#@oO+"), " "),
        lambda: make_box(random.randint(10, 40), random.randint(5, 20), random.choice("#*+=-"), " "),
        lambda: make_triangle(random.randint(5, 20), random.choice("*#@.+")),
        lambda: make_spiral(random.randint(10, 25), random.choice(charset_sets)),
        lambda: make_wave(random.randint(20, 50), random.randint(10, 25), random.choice(charset_sets)),
        lambda: make_random_pattern(random.randint(15, 40), random.randint(8, 20), random.choice(charset_sets), random.uniform(0.15, 0.5)),
        lambda: make_checkerboard(random.randint(10, 30), random.randint(5, 15), random.choice("#@*+"), random.choice(". -")),
    ]

    for i in range(count):
        gen = random.choice(generators)
        art = gen()
        if len(art.strip()) < 10:
            continue
        art_hash = hashlib.md5(art.encode()).hexdigest()[:12]
        filename = f"procedural_{art_hash}.txt"
        filepath = output_dir / filename
        filepath.write_text(art, encoding="utf-8")
        saved.append(str(filepath))

    print(f"  Generated {len(saved)} procedural pieces")
    return saved


def main():
    print(f"Output directory: {RAW_DATA_DIR}")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_files = []

    # Source 1: JSON database from GitHub
    all_files.extend(download_json_db(RAW_DATA_DIR))

    # Source 2: Scrape asciiart.website
    all_files.extend(scrape_asciiart_website(RAW_DATA_DIR))

    # Source 3: Procedural generation for augmentation
    all_files.extend(generate_procedural_art(RAW_DATA_DIR, count=500))

    print(f"\nTotal collected: {len(all_files)} ASCII art pieces")
    print(f"Saved to: {RAW_DATA_DIR}")

    # Write manifest
    manifest_path = RAW_DATA_DIR / "manifest.txt"
    manifest_path.write_text("\n".join(all_files), encoding="utf-8")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
