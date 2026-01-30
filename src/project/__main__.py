from project.simulation import Simulation
from project.ui import Visualization


def main() -> None:
    sim = Simulation(
        name="solar_system_moons",
        horizons=True,
        epoch=(2026, 1, 1),
        dt=3600,
        time=3600 * 24 * 365.25 * 1000,
        # name="figure-8",
        # dt=1e-3,  # simulation time step (seconds)
        # time=6.325,  # simulation time (seconds)
    )
    vis = Visualization(sim=sim)
    vis.start()


if __name__ == "__main__":
    import cProfile
    import os
    import pstats
    import sys
    import tempfile
    import traceback

    if "profile" in sys.argv:
        fd, profile_data_fname = tempfile.mkstemp(suffix=".prof")
        os.close(fd)

        try:
            cProfile.run("main()", profile_data_fname)

            stats = pstats.Stats(profile_data_fname)
            stats.strip_dirs()

            print("\n=== CUMULATIVE TIME ===\n")
            stats.sort_stats("cumulative").print_stats(30)

            print("\n=== PER-FUNCTION TIME ===\n")
            stats.sort_stats("time").print_stats(30)

        finally:
            os.remove(profile_data_fname)
    else:
        try:
            main()
        except Exception:
            traceback.print_exc()
