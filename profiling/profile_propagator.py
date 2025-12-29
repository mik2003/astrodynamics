import cProfile
import pstats

from project.simulation.model import CPPPointMass, NumbaPointMass, NumpyPointMass
from project.simulation.propagator import Propagator
from project.utils import Dir
from project.utils.data import BodyList
from project.utils.time import T


def run_numpy():
    print("Running Numpy...")
    bl = BodyList.load(Dir.data / "solar_system_2460967.toml")

    dt = T.s
    steps = int(T.d)

    p = Propagator("rk4", NumpyPointMass())
    p.propagate(dt, steps, bl)
    print("Numpy done!")


def run_numba():
    print("Running Numba...")
    bl = BodyList.load(Dir.data / "solar_system_2460967.toml")

    dt = T.s
    steps = int(T.d)

    p = Propagator("rk4", NumbaPointMass())
    p.propagate(dt, steps, bl)
    print("Numba done!")


def run_cpp():
    print("Running C++...")
    bl = BodyList.load(Dir.data / "solar_system_2460967.toml")

    dt = T.s
    steps = int(T.d)

    p = Propagator("rk4", CPPPointMass())
    p.propagate(dt, steps, bl)
    print("C++ done!")


def warmup_numba():
    print("Warming up Numba kernel...")
    bl = BodyList.load(Dir.data / "solar_system_2460967.toml")

    kernel = NumbaPointMass()

    state = bl.y_0
    out = state.copy()
    mu = bl.mu
    n = bl.n

    # ONE call is enough
    kernel(state, n, mu, out)

    print("Numba warm-up done!")


if __name__ == "__main__":
    # -------------------------------------------------
    # NUMBA WARM-UP (compile, do NOT profile)
    # -------------------------------------------------
    warmup_numba()

    # -------------------------------------------------
    # PROFILE NUMPY
    # -------------------------------------------------
    profiler = cProfile.Profile()
    profiler.enable()
    run_numpy()
    profiler.disable()
    profiler.dump_stats("profiling/prop_numpy.prof")

    # -------------------------------------------------
    # PROFILE NUMBA (compiled execution only)
    # -------------------------------------------------
    profiler = cProfile.Profile()
    profiler.enable()
    run_numba()
    profiler.disable()
    profiler.dump_stats("profiling/prop_numba.prof")

    # -------------------------------------------------
    # PROFILE CPP (compiled execution only)
    # -------------------------------------------------
    profiler = cProfile.Profile()
    profiler.enable()
    run_cpp()
    profiler.disable()
    profiler.dump_stats("profiling/prop_cpp.prof")

    # -------------------------------------------------
    # QUICK CONSOLE SUMMARY (optional)
    # -------------------------------------------------
    print("Top functions (Numpy):")
    stats = pstats.Stats("profiling/prop_numpy.prof")
    stats.strip_dirs().sort_stats("cumulative").print_stats(15)
    print("Top functions (Numba):")
    stats = pstats.Stats("profiling/prop_numba.prof")
    stats.strip_dirs().sort_stats("cumulative").print_stats(15)
    print("Top functions (CPP):")
    stats = pstats.Stats("profiling/prop_cpp.prof")
    stats.strip_dirs().sort_stats("cumulative").print_stats(15)
