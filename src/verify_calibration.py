"""
Verify calibration data and activation statistics.

Checks:
- All expected samples exist
- Activation statistics are valid
- Data integrity

Usage:
    python -m src.verify_calibration --calib-dir calibration_data
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def verify_calibration_samples(calib_dir: Path, manifest: dict):
    """Verify all calibration samples exist and are valid."""
    
    print("\n=== Verifying Calibration Samples ===")
    
    samples_dir = calib_dir / "samples"
    latents_dir = calib_dir / "latents"
    images_dir = calib_dir / "images"
    
    if not samples_dir.exists():
        print("✗ Samples directory not found")
        return False
    
    num_images = len(manifest['images'])
    num_steps = manifest['num_steps']
    
    # Check samples
    expected_samples = num_images * (num_steps + 1)
    actual_samples = len(list(samples_dir.glob("*.npz")))
    
    print(f"Samples: {actual_samples}/{expected_samples}")
    
    if actual_samples < expected_samples:
        print(f"  ⚠️  Missing {expected_samples - actual_samples} samples")
        
        # Find which images are incomplete
        incomplete = []
        for img_idx in range(num_images):
            for step_idx in range(num_steps + 1):
                sample_file = samples_dir / f"{img_idx:04d}_{step_idx:03d}.npz"
                if not sample_file.exists():
                    incomplete.append((img_idx, step_idx))
        
        if incomplete:
            print(f"  First 10 missing: {incomplete[:10]}")
    else:
        print("  ✓ All samples present")
    
    # Verify sample integrity
    print("\nVerifying sample integrity...")
    
    test_sample = samples_dir / "0000_000.npz"
    if test_sample.exists():
        try:
            data = np.load(test_sample)
            required_keys = ['x', 'timestep', 'sigma', 'step_index', 'image_id']
            
            for key in required_keys:
                if key not in data:
                    print(f"  ✗ Missing key: {key}")
                    return False
            
            print(f"  ✓ Sample format valid")
            print(f"    x shape: {data['x'].shape}")
            print(f"    x dtype: {data['x'].dtype}")
            
        except Exception as e:
            print(f"  ✗ Error loading sample: {e}")
            return False
    
    # Check latents
    if latents_dir.exists():
        expected_latents = num_images
        actual_latents = len(list(latents_dir.glob("*.npy")))
        print(f"\nLatents: {actual_latents}/{expected_latents}")
        
        if actual_latents < expected_latents:
            print(f"  ⚠️  Missing {expected_latents - actual_latents} latents")
        else:
            print("  ✓ All latents present")
    
    # Check images
    if images_dir.exists():
        expected_images = num_images
        actual_images = len(list(images_dir.glob("*.png")))
        print(f"\nImages: {actual_images}/{expected_images}")
        
        if actual_images < expected_images:
            print(f"  ⚠️  Missing {expected_images - actual_images} images")
        else:
            print("  ✓ All images present")
    
    return True


def verify_activation_statistics(calib_dir: Path):
    """Verify activation statistics are valid."""
    
    print("\n=== Verifying Activation Statistics ===")
    
    activations_dir = calib_dir / "activations"
    stats_file = activations_dir / "layer_statistics.json"
    
    if not stats_file.exists():
        print("✗ Activation statistics not found")
        print(f"  Expected: {stats_file}")
        print("  Run collect_layer_activations.py first")
        return False
    
    print(f"✓ Statistics file found: {stats_file}")
    
    # Load and validate
    try:
        with open(stats_file) as f:
            stats = json.load(f)
        
        num_layers = len(stats)
        print(f"✓ Loaded statistics for {num_layers} layers")
        
        # Verify each layer has required fields
        required_fields = ['min', 'max', 'mean', 'std', 'percentiles', 'num_samples']
        
        invalid_layers = []
        for layer_name, layer_stats in stats.items():
            for field in required_fields:
                if field not in layer_stats:
                    invalid_layers.append((layer_name, field))
        
        if invalid_layers:
            print(f"\n✗ Invalid statistics found:")
            for layer, field in invalid_layers[:10]:
                print(f"  {layer}: missing {field}")
            return False
        
        print("✓ All layers have valid statistics")
        
        # Check for suspicious values
        suspicious = []
        for layer_name, layer_stats in stats.items():
            # Check for NaN or inf
            if not np.isfinite(layer_stats['mean']):
                suspicious.append((layer_name, 'non-finite mean'))
            
            # Check for zero range
            if layer_stats['max'] - layer_stats['min'] < 1e-8:
                suspicious.append((layer_name, 'zero range'))
            
            # Check for negative std
            if layer_stats['std'] < 0:
                suspicious.append((layer_name, 'negative std'))
        
        if suspicious:
            print(f"\n⚠️  Suspicious values found:")
            for layer, issue in suspicious[:10]:
                print(f"  {layer}: {issue}")
        else:
            print("✓ No suspicious values detected")
        
        # Print summary
        print("\nStatistics summary:")
        
        all_ranges = [s['max'] - s['min'] for s in stats.values()]
        all_means = [s['mean'] for s in stats.values()]
        all_stds = [s['std'] for s in stats.values()]
        
        print(f"  Range: [{min(all_means):.4f}, {max(all_means):.4f}]")
        print(f"  Mean std: {np.mean(all_stds):.4f}")
        print(f"  Mean range: {np.mean(all_ranges):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading statistics: {e}")
        return False


def estimate_storage(calib_dir: Path):
    """Estimate storage usage."""
    
    print("\n=== Storage Usage ===")
    
    def get_dir_size(directory):
        """Get total size of directory in bytes."""
        total = 0
        for file in directory.rglob("*"):
            if file.is_file():
                total += file.stat().st_size
        return total
    
    components = {
        'samples': calib_dir / "samples",
        'latents': calib_dir / "latents",
        'images': calib_dir / "images",
        'activations': calib_dir / "activations",
    }
    
    total_bytes = 0
    
    for name, directory in components.items():
        if directory.exists():
            size_bytes = get_dir_size(directory)
            size_mb = size_bytes / (1024 * 1024)
            size_gb = size_bytes / (1024 * 1024 * 1024)
            
            if size_gb > 1:
                print(f"{name:15}: {size_gb:6.2f} GB")
            else:
                print(f"{name:15}: {size_mb:6.1f} MB")
            
            total_bytes += size_bytes
    
    total_gb = total_bytes / (1024 * 1024 * 1024)
    print(f"{'Total':15}: {total_gb:6.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Verify calibration data integrity"
    )
    parser.add_argument("--calib-dir", type=Path, required=True,
                       help="Calibration data directory")
    args = parser.parse_args()
    
    if not args.calib_dir.exists():
        print(f"Error: Directory not found: {args.calib_dir}")
        return
    
    print("="*80)
    print("CALIBRATION DATA VERIFICATION")
    print("="*80)
    print(f"\nDirectory: {args.calib_dir}")
    
    # Load manifest
    manifest_path = args.calib_dir / "manifest.json"
    
    if not manifest_path.exists():
        print(f"\n✗ Manifest not found: {manifest_path}")
        print("Run generate_calibration_data.py first")
        return
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print(f"\nManifest:")
    print(f"  Images: {len(manifest['images'])}")
    print(f"  Steps: {manifest['num_steps']}")
    print(f"  Completed: {manifest.get('n_completed', 0)}")
    
    # Run verifications
    samples_ok = verify_calibration_samples(args.calib_dir, manifest)
    activations_ok = verify_activation_statistics(args.calib_dir)
    
    # Estimate storage
    estimate_storage(args.calib_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    if samples_ok:
        print("✓ Calibration samples: PASS")
    else:
        print("✗ Calibration samples: FAIL")
    
    if activations_ok:
        print("✓ Activation statistics: PASS")
    else:
        print("⚠️  Activation statistics: NOT FOUND (run collection script)")
    
    if samples_ok and activations_ok:
        print("\n✓ All checks passed! Data ready for quantization.")
    elif samples_ok:
        print("\n✓ Calibration data ready. Collect activations next:")
        print("  python -m src.collect_layer_activations --calib-dir", args.calib_dir)
    else:
        print("\n✗ Issues found. Check errors above.")
    
    print("="*80)


if __name__ == "__main__":
    main()
