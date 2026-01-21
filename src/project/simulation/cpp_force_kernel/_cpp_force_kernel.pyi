# _cpp_force_kernel.pyi

from project.utils import FloatArray

def point_mass_cpp(
    state: FloatArray,
    mu: FloatArray,
    out: FloatArray,
) -> None: ...
