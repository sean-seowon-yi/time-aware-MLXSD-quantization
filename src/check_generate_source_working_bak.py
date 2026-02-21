"""
Find where generate_image does the sampling.
python -m src.check_generate_source
"""

import sys
from pathlib import Path
import inspect

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from diffusionkit.mlx import DiffusionPipeline

# Get the source of generate_image
source = inspect.getsource(DiffusionPipeline.generate_image)

print("=== generate_image source ===")
print(source)