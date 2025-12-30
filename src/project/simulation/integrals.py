# integrals.py
import time
from pathlib import Path

import numpy as np

from project.utils import FloatArray, print_done, print_progress
from project.utils.simstate import Memmap

G = 6.67430e-11


# ============================================================================
# Internal helpers
# ============================================================================


def _pairwise_potential_energy(
    r: FloatArray,
    mu: FloatArray,
) -> float:
    """
    Compute total gravitational potential energy for one timestep.

    Parameters
    ----------
    r : (3, n) array
        Positions
    mu : (n,) array
        Gravitational parameters (G*m)

    Returns
    -------
    float
        Total potential energy
    """
    n = r.shape[1]
    i_idx, j_idx = np.triu_indices(n, k=1)

    r_i = r[:, i_idx]
    r_j = r[:, j_idx]

    dr = r_i - r_j
    dist = np.linalg.norm(dr, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        U: FloatArray = -(mu[i_idx] * mu[j_idx]) / (G * dist)

    return np.nansum(U)


# ============================================================================
# Energy
# ============================================================================


def calculate_energy(
    sim: Memmap,
    mu: FloatArray,
    cache_file: Path | None = None,
) -> FloatArray:
    """
    Compute total energy time history.

    Parameters
    ----------
    sim : Memmap
        Loaded .simstate file
    mu : (n,) array
        Gravitational parameters (G*m)
    cache_file : Path | None
        Optional .siminteg cache

    Returns
    -------
    (steps,) array
        Total energy per step
    """
    if cache_file and cache_file.exists():
        mm = np.memmap(cache_file, dtype="float64", mode="r", shape=(sim.steps,))
        return np.array(mm)

    n_steps = sim.steps
    n = sim.bodies

    masses = mu / G
    energy = np.zeros(n_steps, dtype=np.float64)

    start_time = time.time()
    print("Calculating total energy...")

    for t in range(n_steps):
        if t % 10000 == 0:
            print_progress(t, n_steps, start_time)

        r = sim.r_vis[t]  # (1,3,n)
        v = sim.v_vis[t]

        r = r[0]
        v = v[0]

        # Kinetic energy
        v2 = np.sum(v * v, axis=0)
        T = 0.5 * np.sum(masses * v2)

        # Potential energy
        U = _pairwise_potential_energy(r, mu)

        energy[t] = T + U

    print_done()

    if cache_file:
        mm = np.memmap(
            cache_file,
            dtype="float64",
            mode="w+",
            shape=(n_steps,),
        )
        mm[:] = energy
        mm.flush()

    return energy


# ============================================================================
# Angular momentum
# ============================================================================


def calculate_angular_momentum(
    sim: Memmap,
    mu: FloatArray,
    cache_file: Path | None = None,
) -> FloatArray:
    """
    Compute total angular momentum vector time history.

    Returns
    -------
    (steps, 3) array
    """
    if cache_file and cache_file.exists():
        mm = np.memmap(cache_file, dtype="float64", mode="r", shape=(sim.steps, 3))
        return np.array(mm)

    n_steps = sim.steps
    masses = mu / G

    h = np.zeros((n_steps, 3), dtype=np.float64)

    print("Calculating angular momentum...")

    for t in range(n_steps):
        r = sim.r_vis[t][0]
        v = sim.v_vis[t][0]

        p = masses[None, :] * v
        h[t] = np.sum(np.cross(r, p, axis=0), axis=1)

    if cache_file:
        mm = np.memmap(
            cache_file,
            dtype="float64",
            mode="w+",
            shape=h.shape,
        )
        mm[:] = h
        mm.flush()

    return h
