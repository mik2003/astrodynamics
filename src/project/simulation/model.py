from typing import Any

import numba as nb
import numpy as np

from project.simulation.cpp_force_kernel import point_mass_cpp
from project.utils import A


class ForceKernel:
    def __call__(self, state: A, *args: Any, **kwargs: Any) -> A:
        raise NotImplementedError


class NumpyPointMass(ForceKernel):
    def __call__(self, state: A, n: int, mu: A, out: A) -> A:
        point_mass_numpy(state, n, mu, out)
        return out


class NumbaPointMass(ForceKernel):
    def __call__(self, state: A, n: int, mu: A, out: A) -> A:
        point_mass_numba(state, n, mu, out)
        return out


class CPPPointMass(ForceKernel):
    def __call__(self, state: A, n: int, mu: A, out: A) -> A:
        return point_mass_cpp(state, mu)


def point_mass_numpy(state: A, n: int, mu: A, out: A) -> None:
    """Point mass gravity (J0)

    Parameters
    ----------
    state : A
        Position and velocity (6N,)

    Returns
    -------
    A
        Velocity and acceleration (6N,)

    Raises
    ------
    ValueError
        If state is not of correct shape
    """
    # Uncomment for debugging
    # if state.shape[0] % 6 != 0:
    #     raise ValueError("State must be of shape (6,)")

    # Initialize output state (if no buffer)
    # out = np.zeros_like(state)

    # r' = v
    out[: 3 * n] = state[3 * n :]

    # Extract positions and calculate distances
    r = state[: 3 * n].reshape(3, n)
    r_ij = r[:, :, np.newaxis] - r[:, np.newaxis, :]

    # Avoid divisions by zero
    dist_sq = np.sum(r_ij**2, axis=0)
    np.fill_diagonal(dist_sq, np.inf)

    # Calculate all accelerations
    dist_cubed = dist_sq * np.sqrt(dist_sq)
    accel_contributions = (
        r_ij * mu[np.newaxis, np.newaxis, :] / dist_cubed[np.newaxis, :, :]
    )

    # v' = a
    out[3 * n :] = np.sum(accel_contributions, axis=2).reshape((3 * n))


@nb.njit(fastmath=True, cache=True)
def point_mass_numba(state: A, n: int, mu: A, out: A) -> None:
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
