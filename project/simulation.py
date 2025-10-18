import math
import os
from pathlib import Path

import numpy as np
from numba import cuda

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
        self.epoch = (
            self.body_list.metadata["epoch"] if self.body_list.metadata else ""
        )

        if not os.path.exists(file_traj):
            # Check if GPU is available
            self.use_gpu = self._check_gpu_availability()
            print(f"Using {'GPU' if self.use_gpu else 'CPU'} acceleration")
            print(f"Simulating {time:.2e} seconds...")
            if self.use_gpu:
                simulate_n_steps_gpu(
                    self.body_list, self.steps, dt, file_traj, prnt=True
                )
            else:
                simulate_n_steps_cpu(
                    self.body_list, self.steps, dt, file_traj, prnt=True
                )
            print("\nSimulation complete.")

        self.mm = np.memmap(
            file_traj,
            dtype="float64",
            mode="r",
            shape=(self.steps, 9, self.num_bodies),
        )

    def _check_gpu_availability(self) -> bool:
        """Check if CUDA GPU is available and appropriate for this problem size"""
        # For small numbers of bodies, CPU is faster
        if self.num_bodies < 50:  # Adjust threshold as needed
            print(
                f"Only {self.num_bodies} bodies - using CPU (faster for small N)"
            )
            return False

        try:
            if not cuda.is_available():
                print("CUDA is not available. Falling back to CPU.")
                return False

            # Your existing GPU checks...
            device = cuda.get_current_device()
            print(f"Found GPU: {device.name}")

            free_mem, total_mem = cuda.current_context().get_memory_info()
            free_mem_gb = free_mem / (1024**3)
            print(
                f"GPU memory: {free_mem_gb:.2f} GB free / {total_mem/(1024**3):.2f} GB total"
            )

            if free_mem_gb < 1.0:
                print("Insufficient GPU memory. Falling back to CPU.")
                return False

            print(f"Using GPU acceleration for {self.num_bodies} bodies")
            return True

        except (RuntimeError, cuda.CudaSupportError) as e:
            print(f"GPU check failed: {e}. Falling back to CPU.")
            return False


def a_cpu(body_list: BodyList, r: A) -> A:
    """
    Compute accelerations for all bodies at given positions r (CPU version).
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


@cuda.jit
def a_gpu_kernel(r, mus, accel, n_bodies):
    """
    GPU kernel to compute accelerations for all bodies.
    Each thread computes acceleration for one body in one dimension.
    """
    # Use cuda.grid(1) for 1D grid
    i = cuda.grid(1)  # pylint: disable=no-value-for-parameter

    if i < n_bodies:
        # Initialize acceleration for body i
        accel_x = 0.0
        accel_y = 0.0
        accel_z = 0.0

        # Position of body i
        r_i_x = r[0, i]
        r_i_y = r[1, i]
        r_i_z = r[2, i]

        # Compute gravitational influence from all other bodies
        for j in range(n_bodies):
            if i != j:  # Skip self-interaction
                # Position difference
                dx = r[0, j] - r_i_x
                dy = r[1, j] - r_i_y
                dz = r[2, j] - r_i_z

                # Squared distance
                dist_sq = dx * dx + dy * dy + dz * dz

                # Avoid division by zero
                if dist_sq > 0.0:
                    # Distance cubed - use math.sqrt instead of np.sqrt
                    dist_cubed = dist_sq * math.sqrt(dist_sq)

                    # Acceleration contribution
                    factor = mus[j] / dist_cubed
                    accel_x += dx * factor
                    accel_y += dy * factor
                    accel_z += dz * factor

        # Store results
        accel[0, i] = accel_x
        accel[1, i] = accel_y
        accel[2, i] = accel_z


def a_gpu(body_list: BodyList, r: A) -> A:
    """
    Compute accelerations for all bodies at given positions r (GPU version).
    r is shaped (3, N), with each column representing a body's position.
    Returns accelerations in the same shape (3, N).
    """
    n_bodies = r.shape[1]

    # Prepare data for GPU
    mus = np.array([body.mu for body in body_list], dtype=np.float64)
    r_gpu = cuda.to_device(r.astype(np.float64))
    mus_gpu = cuda.to_device(mus)
    accel_gpu = cuda.device_array((3, n_bodies), dtype=np.float64)

    # Configure kernel launch - use fewer threads for small numbers of bodies
    threads_per_block = min(256, n_bodies)
    blocks_per_grid = (n_bodies + threads_per_block - 1) // threads_per_block

    # Launch kernel
    a_gpu_kernel[blocks_per_grid, threads_per_block](
        r_gpu, mus_gpu, accel_gpu, n_bodies
    )

    # Copy result back to CPU
    return accel_gpu.copy_to_host()


def rk4_step_cpu(
    body_list: BodyList,
    h: float,
    k_arrays: list,
    buffer: A,
    step: int,
) -> None:

    k_r1, k_r2, k_r3, k_r4, k_v1, k_v2, k_v3, k_v4 = k_arrays

    k_r1[:] = body_list.v_0
    k_v1[:] = a_cpu(body_list, body_list.r_0)

    buffer[step, 0:3, :] = body_list.r_0
    buffer[step, 3:6, :] = k_r1
    buffer[step, 6:9, :] = k_v1

    k_r2[:] = body_list.v_0 + k_v1 * h / 2
    k_v2[:] = a_cpu(body_list, body_list.r_0 + k_r1 * h / 2)

    k_r3[:] = body_list.v_0 + k_v2 * h / 2
    k_v3[:] = a_cpu(body_list, body_list.r_0 + k_r2 * h / 2)

    k_r4[:] = body_list.v_0 + k_v3 * h
    k_v4[:] = a_cpu(body_list, body_list.r_0 + k_r3 * h)

    body_list.r_0 = body_list.r_0 + h / 6 * (k_r1 + 2 * k_r2 + 2 * k_r3 + k_r4)
    body_list.v_0 = body_list.v_0 + h / 6 * (k_v1 + 2 * k_v2 + 2 * k_v3 + k_v4)


def rk4_step_gpu(
    body_list: BodyList,
    h: float,
    k_arrays: list,
    buffer: A,
    step: int,
) -> None:

    k_r1, k_r2, k_r3, k_r4, k_v1, k_v2, k_v3, k_v4 = k_arrays

    k_r1[:] = body_list.v_0
    k_v1[:] = a_gpu(body_list, body_list.r_0)

    buffer[step, 0:3, :] = body_list.r_0
    buffer[step, 3:6, :] = k_r1
    buffer[step, 6:9, :] = k_v1

    k_r2[:] = body_list.v_0 + k_v1 * h / 2
    k_v2[:] = a_gpu(body_list, body_list.r_0 + k_r1 * h / 2)

    k_r3[:] = body_list.v_0 + k_v2 * h / 2
    k_v3[:] = a_gpu(body_list, body_list.r_0 + k_r2 * h / 2)

    k_r4[:] = body_list.v_0 + k_v3 * h
    k_v4[:] = a_gpu(body_list, body_list.r_0 + k_r3 * h)

    body_list.r_0 = body_list.r_0 + h / 6 * (k_r1 + 2 * k_r2 + 2 * k_r3 + k_r4)
    body_list.v_0 = body_list.v_0 + h / 6 * (k_v1 + 2 * k_v2 + 2 * k_v3 + k_v4)


def simulate_n_steps_cpu(
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
        rk4_step_cpu(body_list, dt, k_arrays, buffer, i)
        if prnt and i % 10000 == 0:
            print_progress(i, n, start_time)
    if prnt:
        print_done()
    if filename is not None:
        append_positions_npy(filename, np.array(buffer))


def simulate_n_steps_gpu(
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
        rk4_step_gpu(body_list, dt, k_arrays, buffer, i)
        if prnt and i % 10000 == 0:
            print_progress(i, n, start_time)
    if prnt:
        print_done()
    if filename is not None:
        append_positions_npy(filename, np.array(buffer))
