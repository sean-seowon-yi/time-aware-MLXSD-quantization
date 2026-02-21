"""
Decode saved latents to images using DiffusionKit pipeline.

Usage:
    python -m src.decode_latents --calib-dir calibration_data
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mlx.core as mx
from diffusionkit.mlx import DiffusionPipeline


def decode_latent_to_image(pipeline, latent: mx.array) -> np.ndarray:
    """
    Decode latent to image numpy array.
    
    Args:
        pipeline: DiffusionPipeline instance
        latent: Latent array [1, 16, 64, 64]
    
    Returns:
        Image as numpy array [H, W, 3] in uint8 [0, 255]
    """
    # SD3 VAE scaling factor
    # Latents from the diffusion model need to be scaled before VAE decode
    vae_scaling_factor = 1.5305  # SD3 specific scaling factor
    
    # Scale the latent
    scaled_latent = latent / vae_scaling_factor
    
    # Decode using DiffusionKit's method
    image = pipeline.decode_latents_to_image(scaled_latent)
    image_np = np.array(image)
    
    # Handle batch dimension
    if len(image_np.shape) == 4:
        image_np = image_np[0]
    
    # Handle channel-first [C, H, W] -> [H, W, C]
    if len(image_np.shape) == 3 and image_np.shape[0] in [3, 4]:
        image_np = np.transpose(image_np, (1, 2, 0))
    
    # Normalize to [0, 255]
    if image_np.dtype in [np.float32, np.float64]:
        min_val, max_val = image_np.min(), image_np.max()
        
        if min_val >= -1.1 and max_val <= 1.1:
            # Range [-1, 1]
            image_np = ((image_np + 1.0) / 2.0 * 255).astype(np.uint8)
        elif min_val >= 0 and max_val <= 1.1:
            # Range [0, 1]
            image_np = (image_np * 255).astype(np.uint8)
        else:
            # Normalize
            image_np = ((image_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    
    # Ensure RGB
    if len(image_np.shape) == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[-1] == 4:
        image_np = image_np[:, :, :3]
    
    return image_np


def main():
    parser = argparse.ArgumentParser(description="Decode latents to images")
    parser.add_argument("--calib-dir", type=Path, default=None)
    parser.add_argument("--num-images", type=int, default=None,
                       help="Number of images to decode (default: all)")
    parser.add_argument("--scaling-factor", type=float, default=1.5305,
                       help="VAE scaling factor (default: 1.5305 for SD3)")
    args = parser.parse_args()
    
    calib_dir = args.calib_dir or (_REPO / "calibration_data")
    latents_dir = calib_dir / "latents"
    images_dir = calib_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Load manifest
    manifest_path = calib_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Initialize pipeline
    print("=== Loading Pipeline ===")
    pipeline = DiffusionPipeline(
        shift=3.0,
        use_t5=True,
        model_version="argmaxinc/mlx-stable-diffusion-3-medium",
        low_memory_mode=False,
        a16=True,
        w16=True,
    )
    pipeline.check_and_load_models()
    print(f"✓ Pipeline loaded")
    print(f"Using VAE scaling factor: {args.scaling_factor}")
    
    # Decode all latents
    num_images = args.num_images or len(manifest['images'])
    print(f"\n=== Decoding {num_images} Images ===")
    
    decoded_count = 0
    skipped_count = 0
    
    for img_meta in tqdm(manifest['images'][:num_images]):
        img_id = img_meta['image_id']
        prompt = img_meta.get('prompt', 'unknown')
        
        latent_path = latents_dir / f"{img_id:04d}.npy"
        image_path = images_dir / f"{img_id:04d}.png"
        
        if not latent_path.exists():
            print(f"Warning: Latent {img_id} not found")
            continue
        
        if image_path.exists():
            skipped_count += 1
            continue  # Skip if already exists
        
        # Load and decode
        latent = mx.array(np.load(latent_path))
        image_np = decode_latent_to_image(pipeline, latent)
        
        # Save with metadata
        img = Image.fromarray(image_np)
        
        try:
            from PIL.PngImagePlugin import PngInfo
            metadata = PngInfo()
            metadata.add_text("prompt", prompt)
            metadata.add_text("scaling_factor", str(args.scaling_factor))
            img.save(image_path, pnginfo=metadata)
        except:
            img.save(image_path)
        
        decoded_count += 1
    
    print(f"\n=== Complete ===")
    print(f"Decoded: {decoded_count} images")
    print(f"Skipped: {skipped_count} images (already exist)")
    print(f"Images saved to: {images_dir}")


if __name__ == "__main__":
    main()