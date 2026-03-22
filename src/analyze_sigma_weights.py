#!/usr/bin/env python3
"""
Visualize derivative magnitude across sigma range to guide offset tuning.

This script loads activation statistics and the fitted polynomial schedule,
then plots |dα/dσ| (derivative magnitude) across the denoising timeline for
representative layers. Regions with large derivatives need more optimization
weight, suggesting a smaller offset to emphasize those timesteps.

Usage:
  python src/analyze_sigma_weights.py --stats activation_stats_512.npz \
      --schedule polynomial_clipping_schedule_512_p100.json
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_schedule(schedule_path):
    """Load polynomial schedule and extract coefficients."""
    with open(schedule_path) as f:
        schedule = json.load(f)
    return schedule


def compute_poly_derivative(coeffs, sigma):
    """
    Compute polynomial derivative dα/dσ at given sigma value.

    For polynomial α(σ) = c₀ + c₁σ + c₂σ² + c₃σ³ + ...
    Derivative is: dα/dσ = c₁ + 2c₂σ + 3c₃σ² + ...
    """
    # coeffs are in increasing degree order [c₀, c₁, c₂, ...]
    deriv = np.polyder(coeffs)
    return np.polyval(deriv[::-1], sigma)  # polyval expects decreasing degree order


def select_representative_layers(schedule, n_layers=8):
    """
    Select a diverse set of layers to plot:
    - Different block positions (early, middle, late)
    - Different layer types (attn, mlp)
    - Both image and text streams
    """
    layers = list(schedule['layers'].keys())

    # Organize by characteristics
    candidates = {
        'early_img_attn': [],
        'early_txt_attn': [],
        'middle_img_mlp': [],
        'middle_txt_mlp': [],
        'late_img_attn': [],
        'late_txt_mlp': [],
    }

    for layer_name in layers:
        parts = layer_name.split('_')
        block_num = int(parts[0][2:])  # Extract number from mm0, mm1, etc.
        stream = parts[1]  # 'img' or 'txt'
        proj_type = '_'.join(parts[2:])  # 'attn_k_proj', 'mlp_fc1', etc.

        # Categorize
        if block_num < 8:
            if stream == 'img' and 'attn' in proj_type:
                candidates['early_img_attn'].append(layer_name)
            elif stream == 'txt' and 'attn' in proj_type:
                candidates['early_txt_attn'].append(layer_name)
        elif block_num < 16:
            if stream == 'img' and 'mlp' in proj_type:
                candidates['middle_img_mlp'].append(layer_name)
            elif stream == 'txt' and 'mlp' in proj_type:
                candidates['middle_txt_mlp'].append(layer_name)
        else:
            if stream == 'img' and 'attn' in proj_type:
                candidates['late_img_attn'].append(layer_name)
            elif stream == 'txt' and 'mlp' in proj_type:
                candidates['late_txt_mlp'].append(layer_name)

    # Pick one from each category
    selected = []
    for key in ['early_img_attn', 'early_txt_attn', 'middle_img_mlp', 'middle_txt_mlp', 'late_img_attn', 'late_txt_mlp']:
        if candidates[key]:
            selected.append(candidates[key][0])

    # Add a couple more diverse ones if needed
    if len(selected) < n_layers:
        remaining = [l for l in layers if l not in selected]
        selected.extend(remaining[:n_layers - len(selected)])

    return selected[:n_layers]


def analyze_sigma_weights(schedule_path, output_path='sigma_weight_analysis.png'):
    """
    Generate plots showing derivative magnitude across sigma range.
    """
    schedule = load_schedule(schedule_path)
    sigma_range = schedule['sigma_range']
    sigma_min, sigma_max = sigma_range[0], sigma_range[1]

    # Generate sigma values across the range
    sigmas = np.linspace(sigma_min, sigma_max, 100)

    # Select representative layers
    representative = select_representative_layers(schedule, n_layers=8)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, layer_name in enumerate(representative[:4]):
        ax = axes[idx]
        layer_data = schedule['layers'][layer_name]
        degree = layer_data['degree']
        coeffs = layer_data['coeffs']

        # Compute derivatives across the sigma range
        derivatives = []
        alphas = []
        for sigma in sigmas:
            alpha = np.polyval(coeffs[::-1], sigma)  # Evaluate polynomial
            deriv = compute_poly_derivative(coeffs, sigma)
            derivatives.append(abs(deriv))
            alphas.append(alpha)

        derivatives = np.array(derivatives)
        alphas = np.array(alphas)

        # Plot derivative magnitude
        ax_deriv = ax
        ax_deriv.plot(sigmas, derivatives, 'b-', linewidth=2, label='|dα/dσ|')
        ax_deriv.fill_between(sigmas, 0, derivatives, alpha=0.3, color='blue')
        ax_deriv.set_xlabel('σ (noise level)', fontsize=10)
        ax_deriv.set_ylabel('|dα/dσ| (derivative magnitude)', fontsize=10, color='blue')
        ax_deriv.tick_params(axis='y', labelcolor='blue')
        ax_deriv.set_title(f'{layer_name}\n(degree {degree}, R²={layer_data["r2"]:.3f})', fontsize=10)
        ax_deriv.grid(True, alpha=0.3)

        # Add alpha on secondary y-axis
        ax_alpha = ax_deriv.twinx()
        ax_alpha.plot(sigmas, alphas, 'r--', linewidth=1.5, label='α(σ)', alpha=0.7)
        ax_alpha.set_ylabel('α(σ) (clipping range)', fontsize=10, color='red')
        ax_alpha.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    # Print summary statistics
    print("\n" + "="*70)
    print("DERIVATIVE SENSITIVITY SUMMARY")
    print("="*70)

    for layer_name in representative[:4]:
        layer_data = schedule['layers'][layer_name]
        degree = layer_data['degree']
        coeffs = layer_data['coeffs']

        derivatives = []
        for sigma in sigmas:
            deriv = compute_poly_derivative(coeffs, sigma)
            derivatives.append(abs(deriv))

        derivatives = np.array(derivatives)
        max_deriv = derivatives.max()
        mean_deriv = derivatives.mean()

        # Categorize sensitivity
        if max_deriv < 0.5:
            sensitivity = "LOW (stable, larger offset ok)"
        elif max_deriv < 2.0:
            sensitivity = "MEDIUM (moderate change, offset~1.0)"
        else:
            sensitivity = "HIGH (rapid change, prefer offset<0.5)"

        print(f"\n{layer_name}")
        print(f"  Degree: {degree}")
        print(f"  Max derivative: {max_deriv:.4f}")
        print(f"  Mean derivative: {mean_deriv:.4f}")
        print(f"  → {sensitivity}")


def compare_offset_weights(sigma_max=1.0, sigma_min=0.09):
    """
    Show how different offset values affect the weight ratio.
    """
    print("\n" + "="*70)
    print("OFFSET TUNING GUIDE")
    print("="*70)
    print(f"(Based on σ_max={sigma_max}, σ_min={sigma_min})")
    print()

    offsets = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    print("offset | w(σ_max) | w(σ_min) | ratio | interpretation")
    print("-" * 70)

    for offset in offsets:
        w_max = 1.0 / (sigma_max + offset)
        w_min = 1.0 / (sigma_min + offset)
        ratio = w_min / w_max

        if ratio < 1.5:
            interp = "flat weighting"
        elif ratio < 3:
            interp = "light emphasis on clean steps"
        elif ratio < 6:
            interp = "moderate emphasis (recommended for low σ_max)"
        else:
            interp = "strong emphasis (for high σ_max models)"

        print(f"{offset:5.2f} | {w_max:8.4f} | {w_min:8.4f} | {ratio:5.1f}x | {interp}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', type=str, default='polynomial_clipping_schedule_512_p100.json',
                        help='Path to polynomial schedule JSON')
    parser.add_argument('--output', type=str, default='sigma_weight_analysis.png',
                        help='Output path for visualization')
    parser.add_argument('--sigma-max', type=float, default=1.0,
                        help='Maximum σ in your noise schedule (for offset tuning guide)')
    parser.add_argument('--sigma-min', type=float, default=0.09,
                        help='Minimum σ in your noise schedule')

    args = parser.parse_args()

    # Generate analysis
    analyze_sigma_weights(args.schedule, args.output)

    # Print tuning guide
    compare_offset_weights(sigma_max=args.sigma_max, sigma_min=args.sigma_min)

    print("\n✓ Sensitivity analysis complete.")
