"""
Rebuild manifest.json from existing calibration files.

Useful when generation was interrupted or manifest got corrupted.

Usage:
    python -m src.rebuild_manifest --calib-dir calibration_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def detect_completed_images(calib_dir: Path) -> Dict[int, Dict]:
    """
    Scan calibration directory and detect all completed images.
    
    Returns:
        Dict mapping image_id to metadata
    """
    samples_dir = calib_dir / "samples"
    latents_dir = calib_dir / "latents"
    images_dir = calib_dir / "images"
    
    if not samples_dir.exists():
        print(f"Error: {samples_dir} does not exist")
        return {}
    
    # Find all sample files
    sample_files = list(samples_dir.glob("*.npz"))
    
    if not sample_files:
        print(f"No calibration samples found in {samples_dir}")
        return {}
    
    # Group by image_id
    images_data = {}
    
    for sample_file in sample_files:
        # Parse filename: 0000_025.npz -> image_id=0, step=25
        stem = sample_file.stem  # "0000_025"
        parts = stem.split('_')
        
        if len(parts) != 2:
            continue
        
        try:
            img_id = int(parts[0])
            step_idx = int(parts[1])
        except ValueError:
            continue
        
        if img_id not in images_data:
            images_data[img_id] = {
                'image_id': img_id,
                'steps': set(),
                'has_latent': False,
                'has_image': False,
            }
        
        images_data[img_id]['steps'].add(step_idx)
    
    # Check for latents and images
    for img_id in images_data.keys():
        latent_path = latents_dir / f"{img_id:04d}.npy"
        image_path = images_dir / f"{img_id:04d}.png"
        
        images_data[img_id]['has_latent'] = latent_path.exists()
        images_data[img_id]['has_image'] = image_path.exists()
    
    # Determine number of steps (most common step count)
    step_counts = [len(img['steps']) for img in images_data.values()]
    expected_steps = max(set(step_counts), key=step_counts.count) if step_counts else 51
    
    # Filter to complete images only
    completed = {}
    for img_id, data in images_data.items():
        if len(data['steps']) == expected_steps:
            completed[img_id] = data
        else:
            print(f"Warning: Image {img_id} incomplete: {len(data['steps'])}/{expected_steps} steps")
    
    return completed


def load_prompts_from_csv(csv_path: Path, max_count: int = 10000) -> List[str]:
    """Load prompts from CSV."""
    import csv
    prompts = []
    
    if not csv_path.exists():
        return []
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(prompts) >= max_count:
                break
            p = row.get("prompt", "").strip()
            if p:
                prompts.append(p)
    
    return prompts


def load_sample_metadata(sample_path: Path) -> Dict:
    """Load metadata from first sample of an image."""
    data = np.load(sample_path)
    return {
        'image_id': int(data.get('image_id', 0)),
        'step_index': int(data.get('step_index', 0)),
    }


def rebuild_manifest(calib_dir: Path, prompt_csv: Path, cfg_weight: float = 7.5, 
                    num_steps: int = 50, seed_base: int = 42) -> Dict:
    """
    Rebuild manifest from existing files.
    
    Args:
        calib_dir: Calibration directory
        prompt_csv: CSV with prompts
        cfg_weight: CFG scale used
        num_steps: Number of steps used
        seed_base: Base seed used
    
    Returns:
        Rebuilt manifest dict
    """
    print(f"=== Rebuilding Manifest ===")
    print(f"Calibration dir: {calib_dir}")
    
    # Detect completed images
    print(f"\nScanning for completed images...")
    completed = detect_completed_images(calib_dir)
    
    if not completed:
        print("No completed images found!")
        return None
    
    print(f"Found {len(completed)} completed images: {sorted(completed.keys())}")
    
    # Load prompts
    prompts = load_prompts_from_csv(prompt_csv, max_count=max(completed.keys()) + 100)
    
    if not prompts:
        print(f"\nWarning: Could not load prompts from {prompt_csv}")
        print("Manifest will be created without prompts (prompts can be added later)")
    else:
        print(f"Loaded {len(prompts)} prompts from CSV")
    
    # Build manifest
    images_metadata = []
    
    for img_id in sorted(completed.keys()):
        data = completed[img_id]
        
        # Get prompt if available
        prompt = prompts[img_id] if img_id < len(prompts) else f"[Missing prompt for image {img_id}]"
        
        img_meta = {
            'image_id': img_id,
            'prompt': prompt,
            'seed': seed_base + img_id,
            'cfg_weight': cfg_weight,
            'num_steps': num_steps,
            'filename': f"{img_id:04d}.png",
            'latent_filename': f"{img_id:04d}.npy",
            'steps_found': len(data['steps']),
            'has_latent': data['has_latent'],
            'has_image': data['has_image'],
        }
        
        images_metadata.append(img_meta)
    
    # Create manifest
    manifest = {
        "n_completed": len(completed),
        "num_steps": num_steps,
        "cfg_scale": cfg_weight,
        "latent_size": [64, 64],
        "prompt_path": str(prompt_csv),
        "num_images": max(completed.keys()) + 1,
        "model_version": "argmaxinc/mlx-stable-diffusion-3-medium",
        "use_t5": True,
        "seed_base": seed_base,
        "images_saved": any(img['has_image'] for img in images_metadata),
        "latents_saved": any(img['has_latent'] for img in images_metadata),
        "images": images_metadata,
        "note": "Manifest rebuilt from existing files",
    }
    
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Rebuild manifest.json from existing calibration files")
    parser.add_argument("--calib-dir", type=Path, default=None,
                       help="Calibration directory (default: repo/calibration_data)")
    parser.add_argument("--prompt-csv", type=Path, default=None,
                       help="CSV with prompts (default: repo/all_prompts.csv)")
    parser.add_argument("--cfg-weight", type=float, default=7.5,
                       help="CFG scale used during generation")
    parser.add_argument("--num-steps", type=int, default=50,
                       help="Number of steps used during generation")
    parser.add_argument("--seed-base", type=int, default=42,
                       help="Base seed used during generation")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output path for manifest (default: calib-dir/manifest.json)")
    
    args = parser.parse_args()
    
    calib_dir = args.calib_dir or (_REPO / "calibration_data")
    prompt_csv = args.prompt_csv or (_REPO / "all_prompts.csv")
    output_path = args.output or (calib_dir / "manifest.json")
    
    # Rebuild
    manifest = rebuild_manifest(
        calib_dir,
        prompt_csv,
        cfg_weight=args.cfg_weight,
        num_steps=args.num_steps,
        seed_base=args.seed_base,
    )
    
    if manifest is None:
        print("\nFailed to rebuild manifest")
        return
    
    # Backup existing manifest if it exists
    if output_path.exists():
        backup_path = output_path.with_suffix('.json.backup')
        print(f"\nBacking up existing manifest to {backup_path}")
        import shutil
        shutil.copy(output_path, backup_path)
    
    # Save new manifest
    print(f"\nSaving manifest to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n=== Rebuild Complete ===")
    print(f"Images in manifest: {len(manifest['images'])}")
    print(f"Total steps found: {sum(img['steps_found'] for img in manifest['images'])}")
    print(f"Latents found: {sum(img['has_latent'] for img in manifest['images'])}")
    print(f"Images found: {sum(img['has_image'] for img in manifest['images'])}")


if __name__ == "__main__":
    main()