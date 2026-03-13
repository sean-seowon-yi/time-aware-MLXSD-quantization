"""
Standalone static diagnostic script for W8A8 quantization debugging.

Reads htg_corrections.npz and htg_activation_ranges.npz (no model load required)
and reports three diagnostic sections:

  A — Scaling vector s statistics per layer (detects weight quantization risk)
  B — Activation range summary per layer/group (detects pathological ranges)
  C — Timestep key match check (detects inference vs. calibration mismatch)

Usage
-----
python -m src.inference.diagnose_quantization \\
    --htg-corrections htg_output/htg_corrections.npz \\
    --htg-activation-ranges htg_output/htg_activation_ranges.npz \\
    --num-steps 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _t_key(val: float) -> str:
    return f"{val:.6f}"


def _euler_timesteps(num_steps: int, shift: float = 3.0) -> list[float]:
    """
    Reconstruct DiffusionKit SD3 Euler scheduler timesteps.

        sigma(t) = (shift * t) / (1 + (shift - 1) * t)
        timestep = sigma * 1000

    for t in linspace(1, 0, num_steps + 1), skipping the final 0.
    """
    ts = np.linspace(1.0, 0.0, num_steps + 1, dtype=np.float64)[:-1]
    sigma = (shift * ts) / (1.0 + (shift - 1.0) * ts)
    return (sigma * 1000.0).tolist()


# ---------------------------------------------------------------------------
# Section A — scaling vector s statistics
# ---------------------------------------------------------------------------

def report_s_statistics(corrections: np.lib.npyio.NpzFile) -> None:
    print("\n" + "=" * 72)
    print("Section A — Scaling vector s statistics per layer")
    print("  (fraction > 2.0 indicates potential weight quantization error)")
    print("=" * 72)
    print(f"{'Layer':<40} {'mean':>8} {'min':>8} {'max':>8} {'frac>2':>8}")
    print("-" * 72)

    layer_ids = sorted({
        k.split("::")[0] for k in corrections.files
        if "::" in k and k.split("::")[1] == "s"
    })

    flagged = []
    for lid in layer_ids:
        s = corrections[f"{lid}::s"].astype(np.float32)
        mean_s = float(s.mean())
        min_s  = float(s.min())
        max_s  = float(s.max())
        frac   = float((s > 2.0).mean())
        flag = " ← HIGH" if frac > 0.05 else ""
        print(f"{lid:<40} {mean_s:>8.3f} {min_s:>8.3f} {max_s:>8.3f} {frac:>8.3f}{flag}")
        if frac > 0.05:
            flagged.append(lid)

    if flagged:
        print(f"\n  WARNING: {len(flagged)} layer(s) have >5% channels with s > 2.0")
    else:
        print("\n  OK: all layers have s well-behaved (< 5% channels > 2.0)")


# ---------------------------------------------------------------------------
# Section B — activation range summary
# ---------------------------------------------------------------------------

def report_activation_ranges(ranges_path: str | None) -> None:
    print("\n" + "=" * 72)
    print("Section B — Activation range summary per layer/group")
    print("  (width < 0.1 is pathological; fallback range is (-1.0, 1.0))")
    print("=" * 72)

    if ranges_path is None:
        print("  Skipped (no --htg-activation-ranges provided)")
        return

    raw = np.load(ranges_path, allow_pickle=True)

    # Parse into {layer_id: {g: [min, max]}}
    tmp: dict[str, dict[int, list]] = {}
    for key in raw.files:
        parts = key.split("::")
        if len(parts) != 3:
            continue
        lid, g_tag, stat = parts
        if not g_tag.startswith("g"):
            continue
        g = int(g_tag[1:])
        tmp.setdefault(lid, {}).setdefault(g, [None, None])
        if stat == "act_min":
            tmp[lid][g][0] = float(raw[key])
        elif stat == "act_max":
            tmp[lid][g][1] = float(raw[key])

    print(f"{'Layer':<40} {'groups':>6} {'min_width':>10} {'max_width':>10} {'flags':>20}")
    print("-" * 90)

    pathological_count = 0
    for lid in sorted(tmp):
        gdict = tmp[lid]
        widths = [v[1] - v[0] for v in gdict.values() if v[0] is not None and v[1] is not None]
        if not widths:
            continue
        min_w = min(widths)
        max_w = max(widths)
        flags = []
        if min_w < 0.1:
            flags.append("NARROW")
            pathological_count += 1
        # Check for fallback (-1, 1) range
        for g, (lo, hi) in gdict.items():
            if lo == -1.0 and hi == 1.0:
                flags.append(f"FALLBACK(g{g})")
        flag_str = ", ".join(flags) if flags else "ok"
        print(f"{lid:<40} {len(gdict):>6} {min_w:>10.4f} {max_w:>10.4f} {flag_str:>20}")

    if pathological_count:
        print(f"\n  WARNING: {pathological_count} layer(s) have group width < 0.1")
    else:
        print("\n  OK: all activation ranges are reasonable")


# ---------------------------------------------------------------------------
# Section C — timestep key match check
# ---------------------------------------------------------------------------

def report_timestep_match(corrections: np.lib.npyio.NpzFile, num_steps: int) -> None:
    print("\n" + "=" * 72)
    print(f"Section C — Timestep key match check (num_steps={num_steps}, shift=3.0)")
    print("=" * 72)

    calib_keys = set(_t_key(float(t)) for t in corrections["timesteps_sorted"])
    infer_timesteps = _euler_timesteps(num_steps, shift=3.0)
    infer_keys = [_t_key(t) for t in infer_timesteps]

    matched = [k for k in infer_keys if k in calib_keys]
    unmatched = [k for k in infer_keys if k not in calib_keys]

    print(f"  Calibration timesteps : {len(calib_keys)}")
    print(f"  Inference timesteps   : {len(infer_keys)}")
    print(f"  Matched               : {len(matched)}")
    print(f"  Unmatched             : {len(unmatched)}")

    if unmatched:
        print(f"\n  CRITICAL: {len(unmatched)} inference timestep(s) NOT in calibration set.")
        print("  These will fall back to group 0 ranges — risk of activation clipping!")
        print("\n  Unmatched keys (inference fmt → nearest calib):")
        calib_arr = np.array([float(k) for k in calib_keys])
        for uk in unmatched[:20]:
            uval = float(uk)
            nearest = float(calib_arr[np.argmin(np.abs(calib_arr - uval))])
            print(f"    {uk}  →  nearest calib: {_t_key(nearest)}  (delta={abs(uval - nearest):.6f})")
        if len(unmatched) > 20:
            print(f"    ... and {len(unmatched) - 20} more")
    else:
        print("\n  OK: all inference timesteps found in calibration set.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static quantization diagnostics (no model load required).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--htg-corrections", type=str, required=True,
                        help="Path to htg_corrections.npz")
    parser.add_argument("--htg-activation-ranges", type=str, default=None,
                        help="Path to htg_activation_ranges.npz (optional)")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of inference denoising steps to simulate")
    args = parser.parse_args()

    corrections = np.load(args.htg_corrections, allow_pickle=True)

    report_s_statistics(corrections)
    report_activation_ranges(args.htg_activation_ranges)
    report_timestep_match(corrections, args.num_steps)

    print("\n" + "=" * 72)
    print("Diagnostics complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
