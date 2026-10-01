"""Prompt file loading for benchmark generation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple


def load_prompts(
    prompt_path: Path, max_count: int
) -> Tuple[List[str], Optional[List[int]]]:
    """
    Load up to max_count prompts from CSV ('prompt' column) or plain text.

    Tab-separated .txt with ``seed<TAB>prompt`` returns per-image seeds.
    Missing file → three short synthetic prompts (truncated to max_count).
    """
    if not prompt_path.exists():
        fallback = [
            "a photo of a cat",
            "abstract art with vibrant colors",
            "a landscape with mountains",
        ]
        return fallback[:max_count], None

    prompts: List[str] = []
    seeds: Optional[List[int]] = None

    if prompt_path.suffix.lower() == ".txt":
        with open(prompt_path, encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]

        if lines and "\t" in lines[0]:
            seeds = []
            for line in lines:
                if len(prompts) >= max_count:
                    break
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    try:
                        seeds.append(int(parts[0].strip()))
                        prompts.append(parts[1].strip())
                    except ValueError:
                        pass
        else:
            for line in lines:
                if len(prompts) >= max_count:
                    break
                if line:
                    prompts.append(line)
        return prompts, seeds

    with open(prompt_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(prompts) >= max_count:
                break
            p = row.get("prompt", "").strip()
            if p:
                prompts.append(p)
    return prompts, None
