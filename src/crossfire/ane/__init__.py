"""Apple Neural Engine (ANE) compute target integration.

Provides interfaces to ANEMLL (CoreML path) and Rustane (direct API path)
for running draft models and supplementary inference on the ANE, plus a
direct chunked CoreML harness for Gemma 4 E2B (see `gemma4_chunked`).
"""

from crossfire.ane.gemma4_assets import Gemma4Config
from crossfire.ane.gemma4_chunked import Gemma4ChunkedEngine, GenerationResult

__all__ = [
    "Gemma4ChunkedEngine",
    "Gemma4Config",
    "GenerationResult",
]
