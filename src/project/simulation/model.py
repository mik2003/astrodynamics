"""Force model kernels module"""

from typing import List, cast

import numba as nb
import numpy as np

from project.simulation.cpp_force_kernel import point_mass_cpp
from project.simulation.integrator import FunctionProtocol
from project.utils import FloatArray, ProgressTracker


class NumpyPointMass(FunctionProtocol):
    def __call__(
        self, state: FloatArray, out: FloatArray, n: int, mu: FloatArray
    ) -> None:
        point_mass_numpy(state, n, mu, out)


class NumbaPointMass(FunctionProtocol):
    def __call__(
        self, state: FloatArray, out: FloatArray, n: int, mu: FloatArray
    ) -> None:
        point_mass_numba(state, n, mu, out)

    def _rk4_backend(
        self,
        state: FloatArray,
        time_step: float,
        stop_time: float,
        n: int,
        mu: FloatArray,
        progress: bool = True,
        print_step: int = 10_000,
    ) -> FloatArray:
        steps = int(stop_time / time_step) + 1
        state_buffer = np.empty_like(state)

        if not progress:
            return cast(
                FloatArray, _rk4_numba(state, time_step, steps, n, mu, state_buffer)
            )

        pt = ProgressTracker(
            n=steps, print_step=print_step, name="Integrating Numba RK4"
        )

        out: List[FloatArray] = []

        progress_steps = steps // print_step
        remainder = steps % print_step

        for i in range(progress_steps):
            # First batch (including all)
            if i == 0:
                out.append(
                    _rk4_numba(state, time_step, print_step + 1, n, mu, state_buffer)
                )
            # Successive batches (not including first element)
            else:
                out.append(
                    _rk4_numba(
                        out[-1][-1, :], time_step, print_step + 1, n, mu, state_buffer
                    )[1:, :]
                )

            pt.print(i=i * print_step)

        # Last batch, only if remainder is not zero
        if remainder > 0:
            if progress_steps == 0:
                last_state = state
            else:
                last_state = out[-1][-1, :]

            last = _rk4_numba(
                last_state,
                time_step,
                remainder,
                n,
                mu,
                state_buffer,
            )

            if remainder > 1 and progress_steps != 0:
                out.append(last[1:, :])
            else:
                out.append(last)

            pt.print(i=steps)

        return np.vstack(out)


class CPPPointMass(FunctionProtocol):
    def __call__(
        self, state: FloatArray, out: FloatArray, n: int, mu: FloatArray
    ) -> None:
        point_mass_cpp(state, mu, out)


@nb.njit(fastmath=True, cache=True)
def _rk4_numba(
    state: FloatArray,
    time_step: float,
    steps: int,
    n: int,
    mu: FloatArray,
    out: FloatArray,
) -> FloatArray:
    dim = state.size

    y = np.empty((steps, dim))
    y[0] = state

    k1 = np.empty(dim)
    k2 = np.empty(dim)
    k3 = np.empty(dim)
    k4 = np.empty(dim)

    for i in range(steps - 1):
        point_mass_numba(y[i], n, mu, k1)
        point_mass_numba(y[i] + 0.5 * time_step * k1, n, mu, k2)
        point_mass_numba(y[i] + 0.5 * time_step * k2, n, mu, k3)
        point_mass_numba(y[i] + time_step * k3, n, mu, k4)

        y[i + 1] = y[i] + (time_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return y


def point_mass_numpy(
    state: FloatArray, n: int, mu: FloatArray, out: FloatArray
) -> None:
    """
    Vectorized NumPy point-mass gravity kernel (body-major layout).
    """
    # r' = v
    out[: 3 * n] = state[3 * n :]

    # Positions: (n, 3)
    r = state[: 3 * n].reshape(n, 3)

    # Pairwise displacements: r_j - r_i → (n, n, 3)
    r_i = r[:, None, :]  # (n, 1, 3)
    r_j = r[None, :, :]  # (1, n, 3)
    r_ij = r_j - r_i  # (n, n, 3)

    # Squared distances
    dist_sq = np.sum(r_ij**2, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, np.inf)

    inv_r3 = 1.0 / (dist_sq * np.sqrt(dist_sq))

    # Acceleration: sum over j
    a = np.sum(
        r_ij * (mu[None, :, None] * inv_r3[:, :, None]),
        axis=1,
    )  # (n, 3)

    # Write back interleaved
    out[3 * n :] = a.reshape(3 * n)


@nb.njit(fastmath=True, cache=True)
def point_mass_numba(
    state: FloatArray, n: int, mu: FloatArray, out: FloatArray
) -> None:
    # r' = v
    for k in range(3 * n):
        out[k] = state[k + 3 * n]

    # zero accelerations
    for k in range(3 * n):
        out[k + 3 * n] = 0.0

    # symmetric gravity (serial, no atomics)
    for i in range(n):
        xi = state[3 * i]
        yi = state[3 * i + 1]
        zi = state[3 * i + 2]

        mi = mu[i]

        for j in range(i + 1, n):
            dx = xi - state[3 * j]
            dy = yi - state[3 * j + 1]
            dz = zi - state[3 * j + 2]

            r2 = dx * dx + dy * dy + dz * dz
            inv_r = 1.0 / np.sqrt(r2)
            inv_r3 = inv_r * inv_r * inv_r

            fx = dx * inv_r3
            fy = dy * inv_r3
            fz = dz * inv_r3

            mj = mu[j]

            out[3 * i + 3 * n] -= mj * fx
            out[3 * i + 1 + 3 * n] -= mj * fy
            out[3 * i + 2 + 3 * n] -= mj * fz

            out[3 * j + 3 * n] += mi * fx
            out[3 * j + 1 + 3 * n] += mi * fy
            out[3 * j + 2 + 3 * n] += mi * fz
