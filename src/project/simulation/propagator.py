from pathlib import Path
from typing import Literal

from project.simulation.integrator import Integrator
from project.simulation.model import ForceKernel
from project.utils.cache import simstate_view_from_state_view, write_simstate
from project.utils.data import BodyList


class Propagator:
    def __init__(
        self, integrator: Literal["euler", "rk4"], force_model: ForceKernel
    ) -> None:
        self.integrator = getattr(Integrator, integrator)
        self.force_model = force_model

    def propagate(
        self,
        time_step: float,
        stop_time: float,
        body_list: BodyList,
        filename: Path | None = None,
    ) -> None:
        y = self.integrator(
            body_list.y_0,
            time_step,
            stop_time,
            self.force_model,
            n=body_list.n,
            mu=body_list.mu,
        )  # y.shape = (steps, 6*bodies)

        if filename is not None:
            write_simstate(filename, simstate_view_from_state_view(y, body_list.n))
