"""
Test DiffusionKit's built-in generation.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mlx.core as mx
from diffusionkit.mlx import DiffusionPipeline

# Initialize
pipeline = DiffusionPipeline(
    shift=3.0,
    use_t5=True,
    model_version="argmaxinc/mlx-stable-diffusion-3-medium",
    low_memory_mode=False,
    a16=True,
    w16=True,
)
pipeline.check_and_load_models()

prompt = "a photo of a cat"
seed = 42
num_steps = 50

print("=== Using DiffusionKit's generate_image ===")
mx.random.seed(seed)

result = pipeline.generate_image(
    text=prompt,
    num_steps=num_steps,
    cfg_weight=7.5,
    latent_size=(64, 64),
    seed=seed,
    image_path=None,
    verbose=True,
)

print(f"\nResult type: {type(result)}")
print(f"Result is tuple of length: {len(result) if isinstance(result, tuple) else 'N/A'}")

# Unpack if it's a tuple
if isinstance(result, tuple):
    print(f"Tuple contents:")
    for i, item in enumerate(result):
        print(f"  [{i}]: type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'N/A'}")
    
    # The image is probably the first element
    image = result[0]
    final_latent = result[1] if len(result) > 1 else None
else:
    image = result
    final_latent = None

# Convert and save
image_np = np.array(image)
print(f"\nImage shape: {image_np.shape}")
print(f"Image range: [{image_np.min():.3f}, {image_np.max():.3f}]")

# Handle dimensions
if len(image_np.shape) == 4:
    image_np = image_np[0]

if len(image_np.shape) == 3 and image_np.shape[0] in [3, 4]:
    image_np = np.transpose(image_np, (1, 2, 0))

# Normalize to [0, 255]
if image_np.dtype in [np.float32, np.float64]:
    if image_np.min() >= -1.1 and image_np.max() <= 1.1:
        image_np = ((image_np + 1.0) / 2.0 * 255).astype(np.uint8)
    elif image_np.min() >= 0 and image_np.max() <= 1.1:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = ((image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255).astype(np.uint8)

if image_np.shape[-1] > 3:
    image_np = image_np[:, :3]

Image.fromarray(image_np).save("pipeline_builtin_correct.png")
print(f"\nSaved pipeline_builtin_correct.png")

# Also check if latent was returned
if final_latent is not None:
    print(f"\nFinal latent shape: {final_latent.shape}")
    print(f"Final latent range: [{float(mx.min(final_latent)):.6f}, {float(mx.max(final_latent)):.6f}]")
    print(f"Final latent mean: {float(mx.mean(final_latent)):.6f}")
    print(f"Final latent std: {float(mx.std(final_latent)):.6f}")

print(f"\n{'='*50}")
print(f"CHECK pipeline_builtin_correct.png:")
print(f"✓ If it's a proper cat → DiffusionKit works, our code has a bug")
print(f"✗ If it's noise → Something is wrong with DiffusionKit/model")
print(f"{'='*50}")