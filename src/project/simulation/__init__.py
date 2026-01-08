import os

from project.simulation.model import NumbaPointMass
from project.simulation.propagator import Propagator
from project.utils import Dir
from project.utils.data import BodyList
from project.utils.simstate import SIMSTATE_FILE, SimstateMemmap


class Simulation:
    def __init__(self, name: str, dt: float, time: float) -> None:
        self.name = name
        self.dt = dt
        self.time = time
        self.steps = int(time / dt)

        file_in = Dir.data / (self.name + ".toml")
        file_traj = Dir.simulation / SIMSTATE_FILE.format(self.name, dt, self.steps)

        # Run simulation first and save trajectory with progress tracker
        self.body_list = BodyList.load(file_in)
        self.num_bodies = len(self.body_list)
        if self.body_list.metadata and "epoch" in self.body_list.metadata:
            self.epoch = self.body_list.metadata["epoch"]
        else:
            self.epoch = ""

        if not os.path.exists(file_traj):
            print(f"Simulating {time:.2e} seconds...")
            # simulate_n_steps(self.body_list, self.steps, dt, file_traj, prnt=True)
            p = Propagator("rk4", NumbaPointMass())
            p.propagate(
                time_step=dt,
                stop_time=int(self.steps * dt),
                body_list=self.body_list,
                filename=file_traj,
            )
            print("\nSimulation complete.")

        self.mm = SimstateMemmap(file_traj)
