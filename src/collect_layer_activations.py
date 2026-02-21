"""
Collect layer activations from calibration data for quantization.

Strategy:
- Subsample 100 representative images from 1000
- Collect activations at ~17 key timesteps per image
- Extract statistics for quantization calibration

Based on TaQ-DiT approach: collect per-layer activation statistics
with timestep awareness for optimal quantization.

Usage:
    python -m src.collect_layer_activations \
        --calib-dir calibration_data \
        --num-images 100 \
        --output activations
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

import numpy as np
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mlx.core as mx
from diffusionkit.mlx import DiffusionPipeline, CFGDenoiser


def select_key_timesteps(num_steps: int) -> List[int]:
    """
    Select critical timesteps for activation collection.
    
    Strategy:
    - Always include boundaries (start, end)
    - Dense sampling in middle (where structure forms)
    - Sparse elsewhere
    
    For 50 steps, returns ~17 timesteps
    """
    key_steps = set()
    
    # Boundaries
    key_steps.add(0)
    key_steps.add(num_steps - 1)
    
    # Quarter markers
    key_steps.add(num_steps // 4)
    key_steps.add(num_steps // 2)
    key_steps.add(3 * num_steps // 4)
    
    # Dense in critical region (middle third)
    # This is where structure forms in diffusion models
    for i in range(num_steps // 3, 2 * num_steps // 3, 3):
        key_steps.add(i)
    
    # Sparse in early phase (high noise)
    for i in range(0, num_steps // 3, 5):
        key_steps.add(i)
    
    # Sparse in late phase (refinement)
    for i in range(2 * num_steps // 3, num_steps, 5):
        key_steps.add(i)
    
    return sorted(key_steps)


def select_representative_images(manifest: Dict, num_images: int) -> List[int]:
    """
    Select diverse subset of images.
    
    Uses uniform sampling to get diverse prompts.
    """
    total_images = len(manifest['images'])
    
    if num_images >= total_images:
        return list(range(total_images))
    
    # Uniform sampling
    stride = total_images // num_images
    selected_ids = [i * stride for i in range(num_images)]
    
    return selected_ids


class ActivationCollector:
    """
    Collects activations from DiffusionKit's MMDiT model.
    
    Uses monkey-patching to intercept layer outputs without modifying
    DiffusionKit source code.
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.activations = {}
        self.collection_enabled = False
        self.original_calls = {}
    
    def enable_collection(self):
        """Start collecting activations."""
        self.collection_enabled = True
        self.activations = {}
        self._patch_layers()
    
    def disable_collection(self):
        """Stop collecting and restore original methods."""
        self.collection_enabled = False
        self._unpatch_layers()
    
    def _patch_layers(self):
        """
        Patch model layers to collect activations.
        
        Focuses on key layers for quantization:
        - Transformer blocks (attention + MLP)
        - AdaLN modulation
        """
        model = self.pipeline.mmdit
        
        # Patch multimodal transformer blocks
        if hasattr(model, 'multimodal_transformer_blocks'):
            for i, block in enumerate(model.multimodal_transformer_blocks):
                self._patch_block(block, f'mm_block_{i}')
        
        # Patch unified transformer blocks  
        if hasattr(model, 'unified_transformer_blocks'):
            for i, block in enumerate(model.unified_transformer_blocks):
                self._patch_block(block, f'unified_block_{i}')
    
    def _patch_block(self, block, block_name: str):
        """Patch a transformer block to collect activations."""
        
        # Patch the block's __call__ to capture output
        original_call = block.__call__
        
        if block_name not in self.original_calls:
            self.original_calls[block_name] = original_call
        
        def patched_call(*args, **kwargs):
            output = original_call(*args, **kwargs)
            
            if self.collection_enabled:
                # Store activation (convert to numpy to avoid memory issues)
                if isinstance(output, tuple):
                    # Some blocks return multiple outputs
                    self.activations[block_name] = np.array(output[0])
                else:
                    self.activations[block_name] = np.array(output)
            
            return output
        
        # Apply patch
        block.__call__ = patched_call
    
    def _unpatch_layers(self):
        """Restore original layer methods."""
        model = self.pipeline.mmdit
        
        # Restore multimodal blocks
        if hasattr(model, 'multimodal_transformer_blocks'):
            for i, block in enumerate(model.multimodal_transformer_blocks):
                block_name = f'mm_block_{i}'
                if block_name in self.original_calls:
                    block.__call__ = self.original_calls[block_name]
        
        # Restore unified blocks
        if hasattr(model, 'unified_transformer_blocks'):
            for i, block in enumerate(model.unified_transformer_blocks):
                block_name = f'unified_block_{i}'
                if block_name in self.original_calls:
                    block.__call__ = self.original_calls[block_name]
    
    def collect_for_sample(self, x, timestep, sigma, conditioning, 
                          cfg_weight, pooled):
        """
        Run forward pass and collect activations.
        
        Returns:
            Dict mapping layer names to activation arrays
        """
        self.enable_collection()
        
        try:
            # Run forward pass through denoiser
            denoiser = CFGDenoiser(self.pipeline)
            _ = denoiser(x, timestep, sigma,
                        conditioning=conditioning,
                        cfg_weight=cfg_weight,
                        pooled_conditioning=pooled)
            
            # Return collected activations
            return dict(self.activations)
            
        finally:
            self.disable_collection()


def update_layer_statistics(layer_stats: Dict, layer_name: str, activation: np.ndarray):
    """
    Update running statistics for a layer.
    
    Tracks: min, max, mean, std, percentiles
    """
    stats = layer_stats[layer_name]
    
    # Update range
    stats['min'] = min(stats.get('min', float('inf')), activation.min())
    stats['max'] = max(stats.get('max', float('-inf')), activation.max())
    
    # Sample values for computing statistics
    # (Store subset to avoid memory issues)
    flat = activation.flatten()
    sample_size = min(10000, len(flat))
    if len(flat) > sample_size:
        indices = np.random.choice(len(flat), sample_size, replace=False)
        sampled = flat[indices]
    else:
        sampled = flat
    
    if 'values' not in stats:
        stats['values'] = []
    
    stats['values'].extend(sampled.tolist())
    
    # Limit total stored values to prevent memory issues
    if len(stats['values']) > 100000:
        stats['values'] = stats['values'][-100000:]


def compute_final_statistics(layer_stats: Dict) -> Dict:
    """
    Compute final statistics from collected values.
    
    Returns statistics needed for quantization:
    - Range (min, max)
    - Mean, std
    - Percentiles (for outlier-robust quantization)
    """
    final_stats = {}
    
    for layer_name, stats in layer_stats.items():
        values = np.array(stats['values'])
        
        final_stats[layer_name] = {
            'min': float(stats['min']),
            'max': float(stats['max']),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'percentiles': {
                'p01': float(np.percentile(values, 1)),
                'p05': float(np.percentile(values, 5)),
                'p10': float(np.percentile(values, 10)),
                'p25': float(np.percentile(values, 25)),
                'p50': float(np.percentile(values, 50)),
                'p75': float(np.percentile(values, 75)),
                'p90': float(np.percentile(values, 90)),
                'p95': float(np.percentile(values, 95)),
                'p99': float(np.percentile(values, 99)),
            },
            'num_samples': int(len(values)),
        }
    
    return final_stats


def main():
    parser = argparse.ArgumentParser(
        description="Collect layer activations from calibration data"
    )
    parser.add_argument("--calib-dir", type=Path, required=True,
                       help="Calibration data directory")
    parser.add_argument("--num-images", type=int, default=100,
                       help="Number of representative images to process")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory (default: calib-dir/activations)")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing activation statistics")
    args = parser.parse_args()
    
    output_dir = args.output_dir or (args.calib_dir / "activations")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    stats_file = output_dir / "layer_statistics.json"
    
    # Check if already exists
    if stats_file.exists() and not args.force:
        print(f"Activation statistics already exist at {stats_file}")
        print("Use --force to regenerate")
        return
    
    # Load manifest
    print("=== Loading Calibration Metadata ===")
    manifest_path = args.calib_dir / "manifest.json"
    
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        print("Run generate_calibration_data.py first")
        return
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    num_steps = manifest['num_steps']
    cfg_weight = manifest['cfg_scale']
    
    print(f"Total images in calibration: {len(manifest['images'])}")
    print(f"Steps per image: {num_steps}")
    
    # Select subset
    print("\n=== Selecting Representative Subset ===")
    selected_image_ids = select_representative_images(manifest, args.num_images)
    print(f"Selected {len(selected_image_ids)} images")
    
    # Select key timesteps
    key_timesteps = select_key_timesteps(num_steps)
    print(f"Will collect at {len(key_timesteps)} timesteps: {key_timesteps}")
    
    total_samples = len(selected_image_ids) * len(key_timesteps)
    print(f"Total samples to process: {total_samples}")
    
    # Initialize pipeline
    print("\n=== Initializing Pipeline ===")
    pipeline = DiffusionPipeline(
        shift=3.0,
        use_t5=True,
        model_version="argmaxinc/mlx-stable-diffusion-3-medium",
        low_memory_mode=False,
        a16=True,
        w16=True,
    )
    pipeline.check_and_load_models()
    print("✓ Pipeline loaded")
    
    # Initialize activation collector
    collector = ActivationCollector(pipeline)
    
    # Collect activations
    print("\n=== Collecting Layer Activations ===")
    samples_dir = args.calib_dir / "samples"
    
    layer_stats = defaultdict(lambda: {})
    processed_count = 0
    error_count = 0
    
    start_time = time.time()
    
    with tqdm(total=total_samples, desc="Collecting") as pbar:
        for img_idx in selected_image_ids:
            img_meta = manifest['images'][img_idx]
            prompt = img_meta['prompt']
            
            # Encode prompt once per image
            conditioning, pooled = pipeline.encode_text(prompt, cfg_weight, "")
            mx.eval(conditioning)
            mx.eval(pooled)
            
            for step_idx in key_timesteps:
                # Load calibration sample
                sample_file = samples_dir / f"{img_idx:04d}_{step_idx:03d}.npz"
                
                if not sample_file.exists():
                    pbar.update(1)
                    error_count += 1
                    continue
                
                try:
                    # Load sample
                    sample_data = np.load(sample_file)
                    x = mx.array(sample_data['x'])
                    timestep = mx.array(sample_data['timestep'])
                    sigma = mx.array(sample_data['sigma'])
                    
                    # Collect activations
                    activations = collector.collect_for_sample(
                        x, timestep, sigma, conditioning, cfg_weight, pooled
                    )
                    
                    # Update statistics
                    for layer_name, activation in activations.items():
                        update_layer_statistics(layer_stats, layer_name, activation)
                    
                    processed_count += 1
                    
                except Exception as e:
                    tqdm.write(f"Error on {sample_file.name}: {e}")
                    error_count += 1
                
                pbar.update(1)
    
    elapsed = time.time() - start_time
    
    print(f"\n=== Collection Complete ===")
    print(f"Processed: {processed_count}/{total_samples} samples")
    print(f"Errors: {error_count}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Collected statistics for {len(layer_stats)} layers")
    
    # Compute final statistics
    print("\n=== Computing Final Statistics ===")
    final_stats = compute_final_statistics(layer_stats)
    
    # Display summary
    print("\n=== Layer Statistics Summary ===")
    for layer_name in sorted(final_stats.keys()):
        stats = final_stats[layer_name]
        print(f"\n{layer_name}:")
        print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  P01-P99: [{stats['percentiles']['p01']:.6f}, {stats['percentiles']['p99']:.6f}]")
        print(f"  Samples: {stats['num_samples']:,}")
    
    # Save statistics
    print(f"\n=== Saving Statistics ===")
    with open(stats_file, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    print(f"✓ Saved to {stats_file}")
    
    # Save metadata
    metadata = {
        'num_images_processed': len(selected_image_ids),
        'num_timesteps_per_image': len(key_timesteps),
        'total_samples': processed_count,
        'key_timesteps': key_timesteps,
        'selected_image_ids': selected_image_ids,
        'collection_time_minutes': elapsed / 60,
        'errors': error_count,
    }
    
    metadata_file = output_dir / "collection_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to {metadata_file}")
    
    print("\n" + "="*60)
    print("Activation collection complete!")
    print(f"Statistics available for {len(final_stats)} layers")
    print(f"Ready for quantization calibration")
    print("="*60)


if __name__ == "__main__":
    main()
