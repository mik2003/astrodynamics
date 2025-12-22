from typing import Any

import numba as nb
import numpy as np

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
    for i in range(3 * n):
        out[i] = state[i + 3 * n]

    # positions view
    r = state[: 3 * n].reshape(3, n)

    # zero accelerations
    for i in range(3 * n):
        out[i + 3 * n] = 0.0

    # pairwise gravity
    for i in range(n):
        xi, yi, zi = r[0, i], r[1, i], r[2, i]
        for j in range(n):
            if i == j:
                continue

            dx = xi - r[0, j]
            dy = yi - r[1, j]
            dz = zi - r[2, j]

            r2 = dx * dx + dy * dy + dz * dz
            inv_r3 = 1.0 / (r2 * np.sqrt(r2))

            out[3 * i + 3 * n] -= mu[j] * dx * inv_r3
            out[3 * i + 1 + 3 * n] -= mu[j] * dy * inv_r3
            out[3 * i + 2 + 3 * n] -= mu[j] * dz * inv_r3
