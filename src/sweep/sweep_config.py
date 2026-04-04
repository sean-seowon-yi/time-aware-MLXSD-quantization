"""Sweep configuration: hyperparameter grid, path conventions, and helpers."""

from __future__ import annotations

from pathlib import Path

# ── Fixed parameters (not swept) ──────────────────────────────────────────
BITS = 4
A_BITS = 8
GROUP_SIZE = 64
SWEEP_SEED = 2026

# ── Hyperparameter grid ───────────────────────────────────────────────────
# Each entry overrides qkv_method, alpha, and optionally ssc_tau.
# Per-token A8 (rho_threshold=0.5) is always enabled; it's a fixed policy.
SWEEP_MATRIX: list[dict] = [
    # ── Phase 1: baseline grid (gs64, 3 methods × 3 alpha) ───────────
    {"qkv_method": "max",     "alpha": 0.3},                              # 0
    {"qkv_method": "max",     "alpha": 0.5},                              # 1
    {"qkv_method": "max",     "alpha": 0.7},                              # 2
    {"qkv_method": "geomean", "alpha": 0.3},                              # 3
    {"qkv_method": "geomean", "alpha": 0.5},                              # 4
    {"qkv_method": "geomean", "alpha": 0.7},                              # 5
    {"qkv_method": "l2",      "alpha": 0.3},                              # 6
    {"qkv_method": "l2",      "alpha": 0.5},                              # 7
    {"qkv_method": "l2",      "alpha": 0.7},                              # 8
    # SSC temperature τ=0.5 at α=0.5
    {"qkv_method": "max",     "alpha": 0.5, "ssc_tau": 0.5},             # 9
    {"qkv_method": "geomean", "alpha": 0.5, "ssc_tau": 0.5},             # 10
    {"qkv_method": "l2",      "alpha": 0.5, "ssc_tau": 0.5},             # 11

    # ── Phase 2: refinement (top-3 methods × gs32 + tau probes) ──────
    # gs32 alpha bracket (top-3 methods × α ∈ {0.4, 0.5, 0.6})
    {"qkv_method": "max",     "alpha": 0.4, "group_size": 32},            # 12
    {"qkv_method": "max",     "alpha": 0.5, "group_size": 32},            # 13
    {"qkv_method": "max",     "alpha": 0.6, "group_size": 32},            # 14
    {"qkv_method": "geomean", "alpha": 0.4, "group_size": 32},            # 15
    {"qkv_method": "geomean", "alpha": 0.5, "group_size": 32},            # 16
    {"qkv_method": "geomean", "alpha": 0.6, "group_size": 32},            # 17
    {"qkv_method": "l2",      "alpha": 0.4, "group_size": 32},            # 18
    {"qkv_method": "l2",      "alpha": 0.5, "group_size": 32},            # 19
    {"qkv_method": "l2",      "alpha": 0.6, "group_size": 32},            # 20
    # gs32 + τ=0.8 (combined refinement)
    {"qkv_method": "max",     "alpha": 0.5, "group_size": 32, "ssc_tau": 0.8},  # 21
    {"qkv_method": "geomean", "alpha": 0.5, "group_size": 32, "ssc_tau": 0.8},  # 22
    {"qkv_method": "l2",      "alpha": 0.5, "group_size": 32, "ssc_tau": 0.8},  # 23
    # gs64 + τ=0.8 (isolate tau effect at current group size)
    {"qkv_method": "max",     "alpha": 0.5, "ssc_tau": 0.8},             # 24
    {"qkv_method": "geomean", "alpha": 0.5, "ssc_tau": 0.8},             # 25
    {"qkv_method": "l2",      "alpha": 0.5, "ssc_tau": 0.8},             # 26
]

# ── Path conventions ──────────────────────────────────────────────────────
PROMPTS_FILE = Path("src/settings/evaluation_set.txt")
DIAGNOSTICS_DIR = Path("diagnostics")
QUANTIZED_ROOT = Path("quantized")
RESULTS_ROOT = Path("results")
FP16_SUBDIR = "fp16"
METRICS_ROOT = Path("metrics")


def config_tag(cfg: dict) -> str:
    """Filesystem-safe tag, e.g. ``w4a8_max_a0.50_gs64`` or ``w4a8_max_a0.50_gs64_t0.5``."""
    bits = cfg.get("bits", BITS)
    a_bits = cfg.get("a_bits", A_BITS)
    qkv = cfg["qkv_method"]
    alpha = cfg["alpha"]
    gs = cfg.get("group_size", GROUP_SIZE)
    tau = cfg.get("ssc_tau", 1.0)
    tag = f"w{bits}a{a_bits}_{qkv}_a{alpha:.2f}_gs{gs}"
    if tau != 1.0:
        tag += f"_t{tau:.1f}"
    return tag


def quantized_dir(cfg: dict) -> Path:
    return QUANTIZED_ROOT / config_tag(cfg)


def results_dir(cfg: dict) -> Path:
    return RESULTS_ROOT / config_tag(cfg)


def fp16_dir() -> Path:
    return RESULTS_ROOT / FP16_SUBDIR


def metrics_dir(cfg: dict) -> Path:
    return METRICS_ROOT / config_tag(cfg)


def load_prompt_pairs(path: Path | None = None) -> list[tuple[int, str]]:
    """Load ``(seed, prompt)`` pairs from the evaluation set (tab-separated)."""
    path = path or PROMPTS_FILE
    pairs: list[tuple[int, str]] = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        seed_str, prompt = line.split("\t", 1)
        pairs.append((int(seed_str), prompt.strip()))
    return pairs


def resolve_configs(indices: list[int] | None) -> list[dict]:
    """Return the full matrix or a subset selected by index."""
    if indices is None:
        return list(SWEEP_MATRIX)
    return [SWEEP_MATRIX[i] for i in indices]


def print_matrix() -> None:
    """Pretty-print the sweep matrix for reference."""
    print(f"{'idx':>3}  {'tag':<35}  {'qkv_method':<10}  {'alpha':>5}  {'gs':>4}  {'tau':>5}")
    print("-" * 74)
    for i, cfg in enumerate(SWEEP_MATRIX):
        gs = cfg.get("group_size", GROUP_SIZE)
        tau = cfg.get("ssc_tau", 1.0)
        print(f"{i:>3}  {config_tag(cfg):<35}  {cfg['qkv_method']:<10}  {cfg['alpha']:5.2f}  {gs:4d}  {tau:5.1f}")
