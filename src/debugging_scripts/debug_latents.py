"""
Debug latent saving and loading.
"""

import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mlx.core as mx
from diffusionkit.mlx import DiffusionPipeline

# Load a saved latent
latent_path = Path("calibration_data/latents/0000.npy")

if not latent_path.exists():
    print(f"Error: {latent_path} does not exist")
    sys.exit(1)

print("=== Loading Saved Latent ===")
saved_latent = np.load(latent_path)
print(f"Shape: {saved_latent.shape}")
print(f"Dtype: {saved_latent.dtype}")
print(f"Min: {saved_latent.min():.6f}")
print(f"Max: {saved_latent.max():.6f}")
print(f"Mean: {saved_latent.mean():.6f}")
print(f"Std: {saved_latent.std():.6f}")

# Check if it's all noise (high std relative to mean indicates noise)
if abs(saved_latent.mean()) < 0.1 and saved_latent.std() > 0.5:
    print("\n⚠️  WARNING: Latent looks like pure noise!")
    print("This suggests the denoising process didn't complete properly.")
else:
    print("\n✓ Latent values look reasonable")

# Initialize pipeline and try decode
print("\n=== Testing Decode ===")
pipeline = DiffusionPipeline(
    shift=3.0,
    use_t5=True,
    model_version="argmaxinc/mlx-stable-diffusion-3-medium",
    low_memory_mode=False,
    a16=True,
    w16=True,
)
pipeline.check_and_load_models()

latent_mx = mx.array(saved_latent)

# Try different scaling factors
scaling_factors = [
    ("No scaling", 1.0),
    ("SD3 standard (1.5305)", 1.5305),
    ("SD3 inverse (0.13025)", 0.13025),
    ("SD1.5 (0.18215)", 0.18215),
    ("SD1.5 inverse (5.49)", 1/0.18215),
]

print("\nTrying different scaling factors...")
for name, scale in scaling_factors:
    print(f"\n{name} (scale={scale}):")
    
    scaled = latent_mx / scale
    
    try:
        decoded = pipeline.decode_latents_to_image(scaled)
        decoded_np = np.array(decoded)
        
        # Flatten to check values
        flat = decoded_np.reshape(-1)
        
        print(f"  Output shape: {decoded_np.shape}")
        print(f"  Output dtype: {decoded_np.dtype}")
        print(f"  Output range: [{flat.min():.3f}, {flat.max():.3f}]")
        print(f"  Output mean: {flat.mean():.3f}")
        print(f"  Output std: {flat.std():.3f}")
        
        # Check if it's just noise
        # Real images should have structure (lower std relative to range)
        value_range = flat.max() - flat.min()
        if value_range < 0.1:
            print("  ⚠️  Output is nearly constant (likely all one color)")
        elif flat.std() / value_range > 0.4:
            print("  ⚠️  Output looks like noise (high variance)")
        else:
            print("  ✓ Output has structure (might be a real image!)")
            
            # Save this one to check
            from PIL import Image
            img_np = np.array(decoded)
            if len(img_np.shape) == 4:
                img_np = img_np[0]
            if img_np.shape[0] in [3, 4]:
                img_np = np.transpose(img_np, (1, 2, 0))
            
            if img_np.dtype in [np.float32, np.float64]:
                if img_np.min() >= -1.1 and img_np.max() <= 1.1:
                    img_np = ((img_np + 1.0) / 2.0 * 255).astype(np.uint8)
                elif img_np.min() >= 0 and img_np.max() <= 1.1:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
            
            test_path = Path(f"test_decode_scale_{scale}.png")
            Image.fromarray(img_np[:, :, :3] if img_np.shape[-1] > 3 else img_np).save(test_path)
            print(f"  Saved test image to: {test_path}")
            
    except Exception as e:
        print(f"  ✗ Failed: {e}")