"""
Inspect DiffusionKit's sampler to see how we can hook into it.
python -m src.inspect_sampler
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from diffusionkit.mlx import DiffusionPipeline
import inspect

pipeline = DiffusionPipeline(
    shift=3.0,
    use_t5=True,
    model_version="argmaxinc/mlx-stable-diffusion-3-medium",
    low_memory_mode=False,
    a16=True,
    w16=True,
)

# Check the sampler
print("=== Pipeline Sampler ===")
print(f"Type: {type(pipeline.sampler)}")
print(f"Module: {pipeline.sampler.__class__.__module__}")

# Check what methods it has
print("\nSampler methods:")
for attr in dir(pipeline.sampler):
    if not attr.startswith('_') and callable(getattr(pipeline.sampler, attr)):
        print(f"  - {attr}")
        
# Look at the sample method
if hasattr(pipeline.sampler, 'sample'):
    print(f"\nSample method signature:")
    sig = inspect.signature(pipeline.sampler.sample)
    print(f"  {sig}")
    
# Check the source location
print(f"\nSampler source file:")
print(f"  {inspect.getfile(pipeline.sampler.__class__)}")