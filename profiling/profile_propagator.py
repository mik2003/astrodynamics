import cProfile
import pstats

import numpy as np

from project.simulation.model import CPPPointMass, NumbaPointMass, NumpyPointMass
from project.simulation.propagator import Propagator
from project.utils import Dir, FloatArray
from project.utils.data import BodyList


def propagate_and_return_output(force_model) -> FloatArray:
    bl = BodyList.load(Dir.data / "solar_system_2460967.toml")
    dt = 1.0  # small time step for test
    steps = 1  # single step

    # Create a buffer and propagate
    buffer = np.zeros_like(bl.y_0)
    p = Propagator("rk4", force_model)
    p.integrator(bl.y_0, dt, steps, force_model, n=bl.n, mu=bl.mu, out=buffer)

    return buffer


if __name__ == "__main__":
    # Warm up Numba
    nb_kernel = NumbaPointMass()
    bl = BodyList.load(Dir.data / "solar_system_2460967.toml")
    out = bl.y_0.copy()
    nb_kernel(bl.y_0, bl.n, bl.mu, out)

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
    # Check that results match
    # -------------------------------------------------
    atol = 1e-12  # absolute tolerance
    rtol = 1e-10  # relative tolerance

    print("Numpy vs Numba:", np.allclose(numpy_out, numba_out, rtol=rtol, atol=atol))
    print("Numpy vs C++:", np.allclose(numpy_out, cpp_out, rtol=rtol, atol=atol))

    print("Max abs diff (Numpy vs C++):", np.max(np.abs(numpy_out - cpp_out)))

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
    # Print first 10 entries of each output buffer
    # -------------------------------------------------
    print("\nFirst 10 entries of Numpy output:", numpy_out[:])
    print("First 10 entries of Numba output:", numba_out[:])
    print("First 10 entries of C++ output:", cpp_out[:])
