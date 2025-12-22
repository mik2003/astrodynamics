from pathlib import Path
from typing import Literal

import numpy as np

from project.simulation.integrator import Integrator
from project.simulation.model import ForceKernel
from project.utils import append_positions_npy
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
        buffer = np.zeros((6 * body_list.n,))
        y = self.integrator(
            body_list.y_0,
            time_step,
            stop_time,
            self.force_model,
            n=body_list.n,
            mu=body_list.mu,
            out=buffer,
        )
        if filename is not None:
            append_positions_npy(filename, y)
