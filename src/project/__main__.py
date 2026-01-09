from project.simulation import Simulation
from project.ui import Visualization

if __name__ == "__main__":
    sim = Simulation(
        name="solar_system_dwarf",
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
