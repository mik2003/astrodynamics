from typing import Tuple

from project.simulation.model import NumbaPointMass
from project.simulation.presets import BodySystemPreset
from project.simulation.propagator import Propagator
from project.utils import Dir
from project.utils.data import BodyList
from project.utils.horizons import generate_sim_file
from project.utils.simstate import SIMSTATE_FILE, SimstateMemmap


class Simulation:
    def __init__(
        self,
        name: str,
        dt: float,
        time: float,
        horizons: bool = False,
        epoch: Tuple[int, int, int] | None = None,
    ) -> None:
        if horizons:
            if epoch is None:
                raise ValueError("Must specify epoch for Horizons fetch")
            self.name = f"{name}_{epoch[0]:04d}{epoch[1]:02d}{epoch[2]:02d}"
        else:
            self.name = name
        self.dt = dt
        self.time = time
        self.steps = int(time / dt)

        file_in = Dir.data / (self.name + ".toml")
        file_traj = Dir.simulation / SIMSTATE_FILE.format(self.name, dt, self.steps)

        if not file_in.exists():
            if horizons and epoch is not None and hasattr(BodySystemPreset, name):
                generate_sim_file(
                    name=self.name,
                    bodies=getattr(BodySystemPreset, name),
                    time=epoch,
                )
            else:
                raise ValueError

        if not file_in.exists():
            raise FileNotFoundError

        # Run simulation first and save trajectory with progress tracker
        self.body_list = BodyList.load(file_in)
        self.num_bodies = len(self.body_list)
        if self.body_list.metadata and "epoch" in self.body_list.metadata:
            self.epoch = self.body_list.metadata["epoch"]
        else:
            self.epoch = ""

        if not file_traj.exists():
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
