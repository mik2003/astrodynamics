import cProfile
import pstats

import numpy as np

from project.simulation.model import (
    CPPPointMass,
    ForceKernel,
    NumbaPointMass,
    NumpyPointMass,
)
from project.simulation.propagator import Propagator
from project.utils import Dir, FloatArray
from project.utils.data import BodyList


def propagate_and_return_output(force_model: ForceKernel) -> FloatArray:
    bl = BodyList.load(Dir.data / "solar_system_2460967.toml")

    dt = 1.0  # small timestep
    steps = 3600.0  # integrate to t = 1 s

    p = Propagator("rk4", force_model)

    # Call the real propagation path
    y = p.integrator(
        bl.y_0,
        dt,
        steps,
        force_model,
        n=bl.n,
        mu=bl.mu,
    )

    # Return final state only
    return y[-1].copy()


if __name__ == "__main__":
    # -------------------------------------------------
    # Warm up Numba (compile JIT)
    # -------------------------------------------------
    bl = BodyList.load(Dir.data / "solar_system_2460967.toml")
    nb_kernel = NumbaPointMass()

    warm_out = np.empty_like(bl.y_0)
    nb_kernel(bl.y_0, out=warm_out, n=bl.n, mu=bl.mu)

    # -------------------------------------------------
    # Run and profile
    # -------------------------------------------------
    profiler = cProfile.Profile()
    profiler.enable()
    numpy_out = propagate_and_return_output(NumpyPointMass())
    profiler.disable()
    profiler.dump_stats("profiling/prop_numpy.prof")

    profiler = cProfile.Profile()
    profiler.enable()
    numba_out = propagate_and_return_output(NumbaPointMass())
    profiler.disable()
    profiler.dump_stats("profiling/prop_numba.prof")

    profiler = cProfile.Profile()
    profiler.enable()
    cpp_out = propagate_and_return_output(CPPPointMass())
    profiler.disable()
    profiler.dump_stats("profiling/prop_cpp.prof")

    # -------------------------------------------------
    # Check numerical agreement
    # -------------------------------------------------
    atol = 1e-12
    rtol = 1e-10

    print("Numpy vs Numba:", np.allclose(numpy_out, numba_out, rtol=rtol, atol=atol))
    print("Numpy vs C++:", np.allclose(numpy_out, cpp_out, rtol=rtol, atol=atol))

    print(
        "Max abs diff (Numpy vs C++):",
        np.max(np.abs(numpy_out - cpp_out)),
    )

    # -------------------------------------------------
    # Print top 10 functions from each profiler
    # -------------------------------------------------
    for name, fname in [
        ("Numpy", "profiling/prop_numpy.prof"),
        ("Numba", "profiling/prop_numba.prof"),
        ("CPP", "profiling/prop_cpp.prof"),
    ]:
        print(f"\nTop 10 functions ({name}):")
        stats = pstats.Stats(fname)
        stats.strip_dirs().sort_stats("cumulative").print_stats(10)

    # -------------------------------------------------
    # Print first 10 entries of each output
    # -------------------------------------------------
    print("\nFirst 10 entries of Numpy output:", numpy_out[:10])
    print("First 10 entries of Numba output:", numba_out[:10])
    print("First 10 entries of C++ output:", cpp_out[:10])
