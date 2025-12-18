import os
from pathlib import Path

import numpy as np

from project.data import BodyList
from project.utilities import (
    A,
    Dir,
    append_positions_npy,
    print_done,
    print_progress,
)


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
        if self.body_list.metadata and "epoch" in self.body_list.metadata:
            self.epoch = self.body_list.metadata["epoch"]
        else:
            self.epoch = ""

        if not os.path.exists(file_traj):
            print(f"Simulating {time:.2e} seconds...")
            simulate_n_steps(
                self.body_list, self.steps, dt, file_traj, prnt=True
            )
            print("\nSimulation complete.")

        self.mm = np.memmap(
            file_traj,
            dtype="float64",
            mode="r",
            shape=(self.steps, 9, self.num_bodies),
        )


def a(body_list: BodyList, r: A) -> A:
    """
    Compute accelerations for all bodies at given positions r.
    r is shaped (3, N), with each column representing a body's position.
    Returns accelerations in the same shape (3, N).
    """
    # Precompute all mus once
    mus = np.array([body.mu for body in body_list])

    # Vectorized approach - compute all pairwise differences at once
    # r_i: (3, N, 1), r_j: (3, 1, N) -> r_ij: (3, N, N)
    r_i = r[:, :, np.newaxis]  # Shape: (3, N, 1)
    r_j = r[:, np.newaxis, :]  # Shape: (3, 1, N)
    r_ij = r_j - r_i  # Shape: (3, N, N)

    # Compute squared distances (avoid sqrt)
    dist_sq = np.sum(r_ij**2, axis=0)  # Shape: (N, N)

    # Avoid division by zero and self-interaction
    np.fill_diagonal(dist_sq, np.inf)

    # Compute acceleration contributions
    # mus: (N,) -> (1, N) for broadcasting
    mus_expanded = mus[np.newaxis, :]  # Shape: (1, N)

    # acceleration = mu * r_ij / dist^(3/2)
    # Since we have dist_sq, we use dist_sq^(3/2) = dist_sq * sqrt(dist_sq)
    dist_cubed = dist_sq * np.sqrt(
        dist_sq
    )  # Only one sqrt per pair instead of per component

    # Compute acceleration for each pair: mu * r_ij / dist_cubed
    # r_ij: (3, N, N), mus_expanded: (1, N) -> need to align dimensions
    accel_contributions = (
        r_ij * mus_expanded[np.newaxis, :, :] / dist_cubed[np.newaxis, :, :]
    )

    # Sum over j dimension to get total acceleration for each i
    a_mat = np.sum(accel_contributions, axis=2)  # Shape: (3, N)

    return a_mat



def rk4_step(
    body_list: BodyList,
    h: float,
    k_arrays: list,
    buffer: A,
    step: int,
) -> None:

    k_r1, k_r2, k_r3, k_r4, k_v1, k_v2, k_v3, k_v4 = k_arrays

    k_r1[:] = body_list.v_0
    k_v1[:] = a(body_list, body_list.r_0)

    buffer[step, 0:3, :] = body_list.r_0
    buffer[step, 3:6, :] = k_r1
    buffer[step, 6:9, :] = k_v1

    k_r2[:] = body_list.v_0 + k_v1 * h / 2
    k_v2[:] = a(body_list, body_list.r_0 + k_r1 * h / 2)

    k_r3[:] = body_list.v_0 + k_v2 * h / 2
    k_v3[:] = a(body_list, body_list.r_0 + k_r2 * h / 2)

    k_r4[:] = body_list.v_0 + k_v3 * h
    k_v4[:] = a(body_list, body_list.r_0 + k_r3 * h)

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

    buffer = np.zeros((n, 9, len(body_list)))
    k_arrays = [np.zeros_like(body_list.r_0) for _ in range(8)]
    start_time = time.time()
    for i in range(n):
        rk4_step(body_list, dt, k_arrays, buffer, i)
        if prnt and i % 10000 == 0:
            print_progress(i, n, start_time)
    if prnt:
        print_done()
    if filename is not None:
        append_positions_npy(filename, np.array(buffer))
