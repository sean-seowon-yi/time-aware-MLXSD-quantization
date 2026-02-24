"""
Inspect denoise_latents to understand the sampling loop.
"""

import sys
from pathlib import Path
import inspect

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from diffusionkit.mlx import DiffusionPipeline

# Get the source of denoise_latents
source = inspect.getsource(DiffusionPipeline.denoise_latents)

print("=== denoise_latents source ===")
print(source)