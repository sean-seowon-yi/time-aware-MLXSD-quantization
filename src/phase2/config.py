"""Phase 2 quantization configuration and constants."""

from pathlib import Path

PHASE2_CONFIG = {
    "alpha": 0.5,
    "b_min": 1e-2,
    "b_max": 1e2,
    "w_eps": 1e-12,
    "group_size": 32,
    "bits": 4,
    "a_bits": 8,
    "qkv_method": "l2",
    "ssc_tau": 1.0,
    "per_token_rho_threshold": 0.5,
    "final_layer_bits": 4,
    "exclude_layers": ["context_embedder"],
}

DIAGNOSTICS_DIR = Path("diagnostics")
MODEL_VERSION = "argmaxinc/mlx-stable-diffusion-3-medium"
HIDDEN_SIZE = 1536

DIFFUSIONKIT_SRC = str(Path("DiffusionKit/python/src"))

PIPELINE_KWARGS = {
    "w16": True,
    "shift": 3.0,
    "use_t5": True,
    "low_memory_mode": False,
}

QUANTIZED_WEIGHTS_FILENAME = "mmdit_quantized.safetensors"
QUANTIZE_CONFIG_FILENAME = "quantize_config.json"


def config_tag(cfg: dict, act_quant: str = "dynamic") -> str:
    """Build a short, filesystem-safe tag from key quantization hyperparameters.

    Examples: ``w4a8_max_a0.50_gs64``, ``w4a8_max_a0.50_gs64_t0.3``,
    ``w4a8_max_a0.50_gs64_static``, ``w4a8_max_a0.50_gs64_staticpc``.

    Parameters
    ----------
    act_quant : ``"dynamic"`` | ``"static"``
        When ``"static"``, a ``_static`` (per-tensor) or ``_staticpc``
        (per-channel) suffix is appended.  The granularity is read from
        ``cfg["static_granularity"]`` (default ``"per_tensor"``).
    """
    bits = cfg.get("bits", PHASE2_CONFIG["bits"])
    a_bits = cfg.get("a_bits", PHASE2_CONFIG["a_bits"])
    qkv = cfg.get("qkv_method", PHASE2_CONFIG["qkv_method"])
    alpha = cfg.get("alpha", PHASE2_CONFIG["alpha"])
    gs = cfg.get("group_size", PHASE2_CONFIG["group_size"])
    tau = cfg.get("ssc_tau", PHASE2_CONFIG["ssc_tau"])
    tag = f"w{bits}a{a_bits}_{qkv}_a{alpha:.2f}_gs{gs}"
    if tau != 1.0:
        tag += f"_t{tau:.1f}"
    if act_quant == "static":
        granularity = cfg.get("static_granularity", "per_tensor")
        tag += "_staticpc" if granularity == "per_channel" else "_static"
    return tag


def config_tag_from_meta(meta: dict) -> str:
    """Build the same tag from a saved ``quantize_config.json``."""
    return config_tag(
        {
            "bits": meta.get("bits", PHASE2_CONFIG["bits"]),
            "a_bits": meta.get("a_bits", PHASE2_CONFIG["a_bits"]),
            "qkv_method": meta.get("qkv_method", PHASE2_CONFIG["qkv_method"]),
            "alpha": meta.get("alpha", PHASE2_CONFIG["alpha"]),
            "group_size": meta.get("group_size", PHASE2_CONFIG["group_size"]),
            "ssc_tau": meta.get("ssc_tau", PHASE2_CONFIG["ssc_tau"]),
            "static_granularity": meta.get("static_granularity", "per_tensor"),
        },
        act_quant=meta.get("act_quant", "dynamic"),
    )
