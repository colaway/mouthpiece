"""Mouthpiece: silent lip-reading dictation for macOS."""

import os as _os

# Let PyTorch run ops that MPS lacks on the CPU instead of raising. Must be set before
# torch is imported, which is why it lives here.
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.0.1"
