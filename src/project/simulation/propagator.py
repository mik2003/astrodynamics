"""Propagator module"""

from pathlib import Path
from typing import Literal

from project.simulation.integrator import FunctionProtocol, Integrator
from project.utils.data import BodyList
from project.utils.simstate import simstate_view_from_state_view, write_simstate


class Propagator:
    def __init__(
        self,
        integrator: Literal["euler", "rk4"],
        force_model: FunctionProtocol,
        progress: bool = True,
        print_step: int = 10000,
    ) -> None:
        self.integrator = getattr(Integrator, integrator)
        self.force_model = force_model
        self.progress = progress
        self.print_step = print_step

    def propagate(
        self,
        time_step: float,
        stop_time: float,
        body_list: BodyList,
        filename: Path | None = None,
    ) -> None:
        if self.progress:
            print("Propagating simulation...")
        y = self.integrator(
            body_list.y_0,
            time_step,
            stop_time,
            self.force_model,
            n=body_list.n,
            mu=body_list.mu,
            progress=self.progress,
            print_step=self.print_step,
        )  # y.shape = (steps, 6*bodies)

        if filename is not None:
            if self.progress:
                print("Saving simulation to file...")
            write_simstate(filename, simstate_view_from_state_view(y, body_list.n))

        if self.progress:
            print("Propagation done!")
