import os
from pathlib import Path

import numpy as np

from project.data import BodyList
from project.utilities import A, Dir, append_positions_npy


class Simulation:
    def __init__(self, name: str, dt: float, time: float) -> None:
        self.name = name
        self.dt = dt
        self.time = time
        self.steps = int(time / dt)

        file_in = Dir.data_dir.joinpath(self.name + ".json")
        file_traj = Dir.data_dir.joinpath(f"{self.name}_{dt}_{self.steps}.bin")

        # Run simulation first and save trajectory with progress tracker
        self.body_list = BodyList.load(file_in)
        self.num_bodies = len(self.body_list)
        self.epoch = self.body_list.metadata["epoch"]
        if not os.path.exists(file_traj):
            print(f"Simulating {time:.2e} seconds...")
            simulate_n_steps(
                self.body_list, self.steps, dt, file_traj, prnt=True
            )
            print("\nSimulation complete.")

        self.data = np.memmap(
            file_traj,
            dtype="float64",
            mode="r",
            shape=(self.steps, 9, self.num_bodies),
        )[:, 0:3, :]


def a(body_list: BodyList, r: A) -> A:
    """
    Compute accelerations for all bodies at given positions r.
    r is shaped (3, N), with each column representing a body's position.
    Returns accelerations in the same shape (3, N).
    """
    n_bodies = r.shape[1]
    a_mat = np.zeros_like(r)

    for i in range(n_bodies):
        r_i = r[:, i : i + 1]  # Keep as (3, 1) for broadcasting
        a_i = np.zeros((3, 1))

        for j in range(n_bodies):
            if i == j or body_list[j].mu is None:
                continue

            r_j = r[:, j : j + 1]  # Keep as (3, 1) for broadcasting
            r_ij = r_j - r_i
            dist = np.linalg.norm(r_ij)

            if dist == 0:  # avoid division by zero
                continue

            a_i += body_list[j].mu * r_ij / dist**3

        a_mat[:, i : i + 1] = a_i

    return a_mat


def rk4_step(body_list: BodyList, h: float) -> None:

    k_r1 = body_list.v_0
    k_v1 = a(body_list, body_list.r_0)

    k_r2 = body_list.v_0 + k_v1 * h / 2
    k_v2 = a(body_list, body_list.r_0 + k_r1 * h / 2)

    k_r3 = body_list.v_0 + k_v2 * h / 2
    k_v3 = a(body_list, body_list.r_0 + k_r2 * h / 2)

    k_r4 = body_list.v_0 + k_v3 * h
    k_v4 = a(body_list, body_list.r_0 + k_r3 * h)

    body_list.r_0 = body_list.r_0 + h / 6 * (k_r1 + 2 * k_r2 + 2 * k_r3 + k_r4)
    body_list.v_0 = body_list.v_0 + h / 6 * (k_v1 + 2 * k_v2 + 2 * k_v3 + k_v4)


def simulate_n_steps(
    body_list: BodyList,
    n: int,
    dt: float,
    filename: Path | None = None,
    prnt: bool = False,
) -> None:
    import time

    buffer = []
    start_time = time.time()
    for i in range(n):
        x_0 = np.vstack(
            [body_list.r_0, body_list.v_0, a(body_list, body_list.r_0)]
        )
        buffer.append(x_0)
        rk4_step(body_list, dt)
        if prnt and i % 100 == 0:
            progress = int(i / n * 50)
            bar = "[" + "#" * progress + "-" * (50 - progress) + "]"
            elapsed = time.time() - start_time
            if i > 0:
                est_total = elapsed / (i / n)
                est_remain = est_total - elapsed
                hrs = int(est_remain // 3600)
                mins = int((est_remain % 3600) // 60)
                secs = int(est_remain % 60)
                est_str = f" | ETA: {hrs:02d}:{mins:02d}:{secs:02d}"
            else:
                est_str = ""
            print(f"\rProgress {bar} {i/n*100:.2f}%{est_str}", end="")
    if prnt:
        print(
            "\rProgress [" + "#" * 50 + "] 100.00% | ETA: 00:00:00", end=""
        )  # Show full bar at the end
    if filename is not None:
        append_positions_npy(filename, np.array(buffer))
