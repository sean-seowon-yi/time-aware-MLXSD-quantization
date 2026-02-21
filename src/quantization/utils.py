"""
Utility functions for TaQ-DiT quantization.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import mlx.core as mx


def load_calibration_sample_with_conditioning(
    sample_path: Path,
    manifest: Dict,
    text_encoder_fn,
    cfg_weight: float = 7.5
) -> Dict:
    """
    Load a calibration sample and regenerate its conditioning from the prompt.
    
    Args:
        sample_path: Path to .npz calibration file (e.g., "0000_025.npz")
        manifest: Loaded manifest.json dict (contains prompts)
        text_encoder_fn: Function to encode text (e.g., pipeline.encode_text)
        cfg_weight: CFG scale
    
    Returns:
        Dict with 'x', 'timestep', 'sigma', 'conditioning', 'pooled_conditioning'
    """
    # Load sample
    data = np.load(sample_path)
    image_id = int(data['image_id'])
    
    # Get prompt from manifest
    prompt = manifest['images'][image_id]['prompt']
    
    # Regenerate conditioning
    conditioning, pooled_conditioning = text_encoder_fn(prompt, cfg_weight, "")
    
    return {
        'x': mx.array(data['x']),
        'timestep': mx.array(data['timestep']),
        'sigma': mx.array(data['sigma']),
        'step_index': int(data['step_index']),
        'image_id': image_id,
        'conditioning': conditioning,
        'pooled_conditioning': pooled_conditioning,
        'prompt': prompt,
    }


def load_all_calibration_samples(
    calib_dir: Path,
    text_encoder_fn,
    max_images: int = None,
    step_stride: int = 1,
    cfg_weight: float = 7.5
) -> Tuple[List[Dict], Dict]:
    """
    Load calibration samples with conditioning regenerated from prompts.
    
    Args:
        calib_dir: Calibration directory
        text_encoder_fn: Function to encode text
        max_images: Max number of images to load
        step_stride: Load every Nth step (e.g., 5 = every 5th step)
        cfg_weight: CFG scale
    
    Returns:
        Tuple of (samples_list, manifest_dict)
    """
    # Load manifest
    manifest_path = calib_dir / "manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Load samples
    samples_dir = calib_dir / "samples"
    samples = []
    
    print(f"Loading calibration samples from {samples_dir}")
    print(f"Regenerating conditioning from prompts...")
    
    for img_meta in manifest['images'][:max_images]:
        image_id = img_meta['image_id']
        prompt = img_meta['prompt']
        
        # Encode prompt ONCE per image
        print(f"  Encoding prompt for image {image_id}...", end=" ", flush=True)
        conditioning, pooled_conditioning = text_encoder_fn(prompt, cfg_weight, "")
        print("✓")
        
        # Load all steps for this image
        for step_file in sorted(samples_dir.glob(f"{image_id:04d}_*.npz")):
            step_data = np.load(step_file)
            step_idx = int(step_data['step_index'])
            
            # Apply stride
            if step_idx % step_stride != 0:
                continue
            
            samples.append({
                'x': mx.array(step_data['x']),
                'timestep': mx.array(step_data['timestep']),
                'sigma': mx.array(step_data['sigma']),
                'step_index': step_idx,
                'image_id': image_id,
                'conditioning': conditioning,
                'pooled_conditioning': pooled_conditioning,
                'prompt': prompt,
            })
    
    print(f"Loaded {len(samples)} samples")
    return samples, manifest
