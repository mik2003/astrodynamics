import os
import sys
from pathlib import Path

if sys.platform == "win32":
    runtime_dir = Path(__file__).parent / "_runtime"
    os.add_dll_directory(str(runtime_dir))

from . import _fast_module

# Re-export public API
hello = _fast_module.hello
sum_list = _fast_module.sum_list
add_arrays = _fast_module.add_arrays

__all__ = ["hello", "sum_list", "add_arrays"]
