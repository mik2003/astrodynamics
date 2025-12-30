from typing import Any

import numba as nb
import numpy as np

from project.simulation.cpp_force_kernel import point_mass_cpp
from project.utils import FloatArray


class ForceKernel:
    def __call__(self, state: FloatArray, *args: Any, **kwargs: Any) -> FloatArray:
        raise NotImplementedError


class NumpyPointMass(ForceKernel):
    def __call__(
        self, state: FloatArray, n: int, mu: FloatArray, out: FloatArray
    ) -> FloatArray:
        point_mass_numpy(state, n, mu, out)
        return out


class NumbaPointMass(ForceKernel):
    def __call__(
        self, state: FloatArray, n: int, mu: FloatArray, out: FloatArray
    ) -> FloatArray:
        point_mass_numba(state, n, mu, out)
        return out


class CPPPointMass(ForceKernel):
    def __call__(
        self, state: FloatArray, n: int, mu: FloatArray, out: FloatArray
    ) -> FloatArray:
        point_mass_cpp(state, mu, out)
        return out


def point_mass_numpy(
    state: FloatArray, n: int, mu: FloatArray, out: FloatArray
) -> None:
    """
    Fast vectorized NumPy point-mass gravity kernel.

    Parameters
    ----------
    state : (6*n,) array
        Positions and velocities: [x0,y0,z0,x1,...,vx0,vy0,vz0,...]
    n : int
        Number of bodies
    mu : (n,) array
        Masses of bodies
    out : (6*n,) array
        Output buffer for [v, a] (same layout as state)
    """
    # r' = v
    out[: 3 * n] = state[3 * n :]

    # Positions reshaped
    r = state[: 3 * n].reshape(3, n)

    # Pairwise differences (r_i - r_j)
    r_ij = r[:, :, np.newaxis] - r[:, np.newaxis, :]  # Shape: (3, n, n)

    # Avoid self-interaction
    np.fill_diagonal(r_ij[0], 0.0)
    np.fill_diagonal(r_ij[1], 0.0)
    np.fill_diagonal(r_ij[2], 0.0)

    # Distances cubed
    dist_sq = np.sum(r_ij**2, axis=0)
    np.fill_diagonal(dist_sq, np.inf)
    inv_r3 = 1.0 / (dist_sq * np.sqrt(dist_sq))  # Shape: (n, n)

    # Mass matrix
    mu_j = mu[np.newaxis, :]  # Shape: (1, n)

    # Acceleration contributions
    contrib = r_ij * (mu_j * inv_r3)  # Shape: (3, n, n)

    # Symmetric sum: subtract for i, add for j
    out[3 * n :] = np.sum(contrib - contrib.transpose(0, 2, 1), axis=2).reshape(3 * n)


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
