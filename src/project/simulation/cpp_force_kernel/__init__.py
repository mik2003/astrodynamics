import os
import sys
from pathlib import Path

# Ensure Windows can find DLLs if running on Win32
if sys.platform == "win32":
    runtime_dir = Path(__file__).parent / "_runtime"
    if runtime_dir.exists():
        os.add_dll_directory(str(runtime_dir))

# Import the compiled extension
from . import _cpp_force_kernel

# Re-export the public API
point_mass_cpp = _cpp_force_kernel.point_mass_cpp

__all__ = ["point_mass_cpp"]
