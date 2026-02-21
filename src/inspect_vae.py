"""
Inspect DiffusionKit pipeline to find VAE decode methods.
python -m src.inspect_vae
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from diffusionkit.mlx import DiffusionPipeline
import mlx.core as mx

print("=== Initializing Pipeline ===")
pipeline = DiffusionPipeline(
    shift=3.0,
    use_t5=True,
    model_version="argmaxinc/mlx-stable-diffusion-3-medium",
    low_memory_mode=False,
    a16=True,
    w16=True,
)
pipeline.check_and_load_models()

print("\n=== Pipeline Attributes (decode-related) ===")
for attr in dir(pipeline):
    if 'decode' in attr.lower() or 'vae' in attr.lower() or 'image' in attr.lower():
        print(f"  - pipeline.{attr}")
        obj = getattr(pipeline, attr)
        if not callable(obj):
            print(f"      Type: {type(obj)}")

print("\n=== Checking VAE Object ===")
if hasattr(pipeline, 'vae'):
    print("✓ pipeline.vae exists")
    print(f"  Type: {type(pipeline.vae)}")
    print("\n  VAE methods:")
    for attr in dir(pipeline.vae):
        if not attr.startswith('_'):
            print(f"    - vae.{attr}")
            obj = getattr(pipeline.vae, attr)
            if callable(obj):
                print(f"        (callable)")
else:
    print("✗ pipeline.vae does not exist")

print("\n=== Testing Decode ===")
# Create dummy latent
dummy_latent = mx.random.normal(shape=(1, 16, 64, 64))

# Try different methods
methods_to_try = [
    ('pipeline.vae.decode', lambda: pipeline.vae.decode(dummy_latent)),
    ('pipeline.vae()', lambda: pipeline.vae(dummy_latent)),
    ('pipeline.decode', lambda: pipeline.decode(dummy_latent)),
    ('pipeline.latent_to_image', lambda: pipeline.latent_to_image(dummy_latent)),
]

for name, func in methods_to_try:
    try:
        print(f"\nTrying {name}...", end=" ")
        result = func()
        print(f"✓ SUCCESS")
        print(f"  Output shape: {result.shape}")
        print(f"  Output dtype: {result.dtype}")
        print(f"  Output range: [{float(mx.min(result)):.3f}, {float(mx.max(result)):.3f}]")
        break
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {str(e)[:80]}")