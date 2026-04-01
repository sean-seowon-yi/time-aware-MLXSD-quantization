import json
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ---------------------------------------------------------------------------
# Load all benchmark.json files
# ---------------------------------------------------------------------------
BASE = pathlib.Path("benchmark_results")

DISPLAY_NAMES = {
    "fp16":                                    "FP16 (baseline)",
    "w4a8_rtn":                                "W4A8 RTN",
    "w4a8_adaround_static_p100":               "W4A8 AdaRound Static",
    "w4a8_adaround_poly_p100":                 "W4A8 AdaRound Poly",
    "w4a8_adaround_poly_p100_deriv":           "W4A8 AdaRound Poly + Deriv",
    "w4a8_adaround_poly_p100_deriv_sigma_offset1": "W4A8 Poly + Deriv + σ-offset1",
    "w4a8_adaround_poly_p100_sigma_offset01":  "W4A8 Poly + σ-offset0.1",
    "w4a8_adaround_poly_p100_group64":         "W4A8 Poly + Group64",
    "w4a8_adaround_poly_p100_group64_derivmax":"W4A8 Poly + Group64 + DerivMax",
    "w4a8_adaround_poly_p100_group64_derivmean":"W4A8 Poly + Group64 + DerivMean",
}

ROW_ORDER = [
    "fp16",
    "w4a8_rtn",
    "w4a8_adaround_static_p100",
    "w4a8_adaround_poly_p100",
    "w4a8_adaround_poly_p100_deriv",
    "w4a8_adaround_poly_p100_deriv_sigma_offset1",
    "w4a8_adaround_poly_p100_sigma_offset01",
    "w4a8_adaround_poly_p100_group64",
    "w4a8_adaround_poly_p100_group64_derivmax",
    "w4a8_adaround_poly_p100_group64_derivmean",
]

def _get(d, *keys, fmt=None, default="—"):
    v = d
    for k in keys:
        if v is None or not isinstance(v, dict):
            return default
        v = v.get(k)
    if v is None:
        return default
    return fmt % v if fmt else v

rows = []
for key in ROW_ORDER:
    p = BASE / key / "benchmark.json"
    if not p.exists():
        continue
    with open(p) as f:
        d = json.load(f)
    fid   = _get(d, "fidelity", "fid",       fmt="%.1f")
    cmmd  = _get(d, "fidelity", "cmmd",      fmt="%.4f")
    sfid  = _get(d, "fidelity", "sfid",      fmt="%.3f")
    kid   = _get(d, "fidelity", "kid_mean",  fmt="%.4f")
    psnr  = _get(d, "paired",   "psnr_mean", fmt="%.2f")
    lpips = _get(d, "paired",   "lpips_mean",fmt="%.3f")
    lat   = _get(d, "latency",  "mean_s",    fmt="%.1f")
    rows.append([DISPLAY_NAMES.get(key, key), fid, cmmd, sfid, kid, psnr, lpips, lat])

# ---------------------------------------------------------------------------
# Find best (lowest) value per numeric column for highlighting
# ---------------------------------------------------------------------------
COLS = ["Config", "FID ↓", "CMMD ↓", "sFID ↓", "KID ↓", "PSNR ↑\n(vs FP16)", "LPIPS ↓\n(vs FP16)", "Latency\n(s)"]
LOWER_BETTER = [True, True, True, True, False, True, True]  # per metric col

def numeric(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

best = []
for ci, lb in enumerate(LOWER_BETTER):
    vals = [numeric(r[ci + 1]) for r in rows]
    valid = [v for v in vals if v is not None]
    if not valid:
        best.append(None)
    elif lb:
        best.append(min(valid))
    else:
        best.append(max(valid))

# ---------------------------------------------------------------------------
# Draw table
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, len(rows) * 0.55 + 1.2))
ax.axis("off")

cell_text = [[r[0]] + [str(v) for v in r[1:]] for r in rows]
col_widths = [0.28] + [0.09] * (len(COLS) - 1)

tbl = ax.table(
    cellText=cell_text,
    colLabels=COLS,
    cellLoc="center",
    loc="center",
    colWidths=col_widths,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)

# Style header
for j in range(len(COLS)):
    cell = tbl[0, j]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")

# Style data rows — alternate shading + highlight best
for i, row in enumerate(rows):
    base_color = "#f0f4f8" if i % 2 == 0 else "#ffffff"
    # Grey out fp16 row (no fidelity metrics)
    is_baseline = row[0] == "FP16 (baseline)"
    for j in range(len(COLS)):
        cell = tbl[i + 1, j]
        cell.set_facecolor("#e8e8e8" if is_baseline else base_color)
        # Highlight best value in each metric column
        if j > 0:
            ci = j - 1
            v = numeric(rows[i][j])
            if v is not None and best[ci] is not None and v == best[ci]:
                cell.set_facecolor("#d4edda")
                cell.set_text_props(fontweight="bold", color="#155724")

fig.suptitle("SD3 Quantization — Benchmark Summary (256 images, 30 steps, CFG 4.0)",
             fontsize=12, fontweight="bold", y=0.98)
fig.text(0.5, 0.01, "Green = best per column. FP16 is reference (no fidelity metrics). PSNR/LPIPS vs FP16 baseline.",
         ha="center", fontsize=8, color="#555")

plt.tight_layout()
out = "benchmark_results/summary_table.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
