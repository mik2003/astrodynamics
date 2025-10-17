from project.simulation import Simulation
from project.visualization import Visualization

if __name__ == "__main__":
    sim = Simulation(
        name="sun_earth_moon_2460966",
        dt=3600,  # simulation time step (seconds)
        time=3600 * 24 * 365.25 * 100,  # simulation time (seconds)
    )
    vis = Visualization(
        sim=sim,
        trail_step_time=3600,  # [s]
        trail_time=3600 * 24 * 365.25,  # [s]
    )
    vis.start()
