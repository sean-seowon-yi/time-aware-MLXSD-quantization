"""
Analyze collected activation statistics.

Helps identify:
- Which layers need more bits
- Timestep-dependent quantization strategies
- Outlier handling approaches

Usage:
    python -m src.analyze_activations --stats calibration_data/activations/layer_statistics.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def analyze_dynamic_range(stats: Dict) -> Dict:
    """
    Analyze dynamic range of each layer.
    
    Larger range = needs more bits or careful scaling
    """
    ranges = {}
    
    for layer_name, layer_stats in stats.items():
        full_range = layer_stats['max'] - layer_stats['min']
        p99_range = layer_stats['percentiles']['p99'] - layer_stats['percentiles']['p01']
        
        # Outlier ratio: how much do outliers affect range?
        outlier_ratio = p99_range / full_range if full_range > 0 else 1.0
        
        ranges[layer_name] = {
            'full_range': full_range,
            'p99_range': p99_range,
            'outlier_ratio': outlier_ratio,
        }
    
    return ranges


def suggest_bit_widths(stats: Dict) -> Dict:
    """
    Suggest bit-widths for each layer based on activation statistics.
    
    Heuristics:
    - Large dynamic range → more bits needed
    - High outlier ratio → use percentile-based clipping
    - High std relative to range → more bits needed
    """
    suggestions = {}
    
    ranges = analyze_dynamic_range(stats)
    
    for layer_name, layer_stats in stats.items():
        range_info = ranges[layer_name]
        
        # Compute relative std (coefficient of variation)
        rel_std = layer_stats['std'] / (range_info['p99_range'] + 1e-8)
        
        # Suggest bits based on range and variation
        if range_info['p99_range'] > 10:
            suggested_bits = 8
        elif range_info['p99_range'] > 5:
            suggested_bits = 6
        elif range_info['p99_range'] > 2:
            suggested_bits = 4
        else:
            suggested_bits = 4
        
        # Adjust for high variation
        if rel_std > 0.5:
            suggested_bits = min(8, suggested_bits + 2)
        
        # Recommend clipping if many outliers
        use_clipping = range_info['outlier_ratio'] < 0.9
        
        suggestions[layer_name] = {
            'suggested_weight_bits': suggested_bits,
            'suggested_activation_bits': suggested_bits,
            'use_percentile_clipping': use_clipping,
            'clip_percentile': (1, 99) if use_clipping else None,
        }
    
    return suggestions


def identify_sensitive_layers(stats: Dict, suggestions: Dict) -> list:
    """
    Identify layers that are most sensitive to quantization.
    
    These layers should get higher bit-widths or special handling.
    """
    sensitive = []
    
    for layer_name in stats.keys():
        sugg = suggestions[layer_name]
        
        # Layers needing 8-bit are sensitive
        if sugg['suggested_activation_bits'] >= 8:
            sensitive.append({
                'layer': layer_name,
                'reason': 'large_dynamic_range',
                'suggested_bits': sugg['suggested_activation_bits'],
            })
    
    return sensitive


def print_summary(stats: Dict):
    """Print comprehensive summary of activation statistics."""
    
    print("\n" + "="*80)
    print("ACTIVATION STATISTICS SUMMARY")
    print("="*80)
    
    print(f"\nTotal layers analyzed: {len(stats)}")
    
    # Overall statistics
    all_mins = [s['min'] for s in stats.values()]
    all_maxs = [s['max'] for s in stats.values()]
    all_stds = [s['std'] for s in stats.values()]
    
    print(f"\nOverall activation range: [{min(all_mins):.4f}, {max(all_maxs):.4f}]")
    print(f"Mean std across layers: {np.mean(all_stds):.4f}")
    
    # Analyze dynamic ranges
    print("\n" + "-"*80)
    print("DYNAMIC RANGE ANALYSIS")
    print("-"*80)
    
    ranges = analyze_dynamic_range(stats)
    
    # Sort by range
    sorted_layers = sorted(ranges.items(), 
                          key=lambda x: x[1]['full_range'], 
                          reverse=True)
    
    print("\nTop 10 layers by dynamic range:")
    for i, (layer_name, range_info) in enumerate(sorted_layers[:10], 1):
        print(f"{i:2}. {layer_name:40} "
              f"Range: {range_info['full_range']:8.4f}  "
              f"P99: {range_info['p99_range']:8.4f}  "
              f"Outlier ratio: {range_info['outlier_ratio']:.2f}")
    
    # Suggest bit-widths
    print("\n" + "-"*80)
    print("QUANTIZATION RECOMMENDATIONS")
    print("-"*80)
    
    suggestions = suggest_bit_widths(stats)
    
    # Count by suggested bits
    bit_counts = {}
    for sugg in suggestions.values():
        bits = sugg['suggested_activation_bits']
        bit_counts[bits] = bit_counts.get(bits, 0) + 1
    
    print("\nSuggested bit-width distribution:")
    for bits in sorted(bit_counts.keys()):
        count = bit_counts[bits]
        pct = 100 * count / len(suggestions)
        print(f"  {bits}-bit: {count:3} layers ({pct:5.1f}%)")
    
    # Identify sensitive layers
    sensitive = identify_sensitive_layers(stats, suggestions)
    
    if sensitive:
        print(f"\nSensitive layers requiring careful quantization ({len(sensitive)}):")
        for item in sensitive[:10]:  # Show top 10
            print(f"  - {item['layer']:40} ({item['reason']}, needs {item['suggested_bits']}-bit)")
    
    # Layers that can use aggressive quantization
    aggressive = [name for name, sugg in suggestions.items() 
                 if sugg['suggested_activation_bits'] <= 4]
    
    if aggressive:
        print(f"\nLayers suitable for aggressive quantization (≤4-bit): {len(aggressive)}")
        for layer in aggressive[:5]:
            print(f"  - {layer}")
    
    print("\n" + "="*80)


def export_quantization_config(stats: Dict, output_path: Path):
    """
    Export quantization configuration for use in quantization scripts.
    
    Format compatible with TaQ-DiT style quantization.
    """
    suggestions = suggest_bit_widths(stats)
    
    config = {
        'layer_configs': {},
        'default_config': {
            'weight_bits': 4,
            'activation_bits': 8,
            'use_symmetric': True,
        }
    }
    
    for layer_name, sugg in suggestions.items():
        config['layer_configs'][layer_name] = {
            'weight_bits': sugg['suggested_weight_bits'],
            'activation_bits': sugg['suggested_activation_bits'],
            'clip_percentile': sugg['clip_percentile'],
            'calibration_stats': {
                'min': stats[layer_name]['min'],
                'max': stats[layer_name]['max'],
                'mean': stats[layer_name]['mean'],
                'std': stats[layer_name]['std'],
                'p01': stats[layer_name]['percentiles']['p01'],
                'p99': stats[layer_name]['percentiles']['p99'],
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Exported quantization config to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze activation statistics"
    )
    parser.add_argument("--stats", type=Path, required=True,
                       help="Path to layer_statistics.json")
    parser.add_argument("--export-config", type=Path, default=None,
                       help="Export quantization config to JSON")
    args = parser.parse_args()
    
    if not args.stats.exists():
        print(f"Error: Statistics file not found: {args.stats}")
        print("Run collect_layer_activations.py first")
        return
    
    # Load statistics
    print(f"Loading statistics from {args.stats}")
    with open(args.stats) as f:
        stats = json.load(f)
    
    # Print summary
    print_summary(stats)
    
    # Export config if requested
    if args.export_config:
        export_quantization_config(stats, args.export_config)


if __name__ == "__main__":
    main()
