from typing import Optional
import os

CACHE_DIR: Optional[str] = os.environ.get("HF_KERNELS_CACHE", None)
