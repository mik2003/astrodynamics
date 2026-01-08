import time
from pathlib import Path

import numba as nb
import numpy as np

from project.utils import Float, FloatArray, print_done, print_progress
from project.utils.siminteg import SIMINTEG_FILE, SimintegMemmap, write_siminteg
from project.utils.simstate import SimstateMemmap

G = 6.67430e-11


# ============================================================================
# Internal helpers
# ============================================================================


def _pairwise_potential_energy(r: FloatArray, mu: FloatArray) -> Float:
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


@nb.njit(fastmath=True, cache=True)
def _pairwise_potential_energy_numba(r: FloatArray, mu: FloatArray, G: float) -> Float:
    n = r.shape[1]
    U = 0.0

    for i in range(n):
        xi = r[0, i]
        yi = r[1, i]
        zi = r[2, i]

        mui = mu[i]

        for j in range(i + 1, n):
            dx = xi - r[0, j]
            dy = yi - r[1, j]
            dz = zi - r[2, j]

            r2 = dx * dx + dy * dy + dz * dz
            inv_r = 1.0 / np.sqrt(r2)

            U -= mui * mu[j] * inv_r / G

    return U


# ============================================================================
# Main integrals calculation
# ============================================================================


def calculate_integrals(
    sim: SimstateMemmap,
    mu: FloatArray,
    cache_file: Path | None = None,
    verbose: bool = True,
) -> FloatArray:
    """
    Compute total energy and angular momentum time history.

    Parameters
    ----------
    sim : SimstateMemmap
        Loaded .simstate file
    mu : (n,) array
        Gravitational parameters (G*m)
    cache_file : Path | None
        Optional .siminteg cache file
    verbose : bool
        Print progress

    Returns
    -------
    integrals : (steps, 4) array
        Column 0: total energy
        Columns 1-3: angular momentum vector
    """
    n_steps = sim.steps
    masses = mu / G

    # Use cache if available
    if cache_file and cache_file.exists():
        if verbose:
            print(f"Loading integrals from cache: {cache_file}")
        mm = np.memmap(cache_file, dtype="float64", mode="r", shape=(n_steps, 4))
        return np.array(mm)

    integrals = np.zeros((n_steps, 4), dtype=np.float64)

    start_time = time.time()
    if verbose:
        print("Calculating energy and angular momentum...")

    for t in range(n_steps):
        if verbose and t % 10000 == 0:
            print_progress(t, n_steps, start_time)

        # Positions & velocities (shape: 3 x n)
        r = sim.r_vis[t][0]
        v = sim.v_vis[t][0]

        # Kinetic energy
        T = 0.5 * np.sum(masses * np.sum(v**2, axis=0))

        # Potential energy
        U = _pairwise_potential_energy_numba(r, mu, G)

        # Total energy
        integrals[t, 0] = T + U

        # Angular momentum vector
        p = masses[None, :] * v
        integrals[t, 1:] = np.sum(np.cross(r, p, axis=0), axis=1)

    if verbose:
        print_done()

    # Save to cache
    if cache_file:
        if verbose:
            print(f"\nSaving integrals to cache: {cache_file}")
        write_siminteg(cache_file, integrals)

    return integrals


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from project.simulation import Simulation
    from project.utils import Dir

    # Load simulation
    sim = Simulation(
        name="solar_system_2460967",
        dt=3600,
        time=3600 * 24 * 365.25 * 1000,
    )

    mu_list = [body.mu for body in sim.body_list]
    mu_arr = np.array(mu_list)

    # Create cache filenames based on simulation parameters
    integ_cache = Dir.simulation / SIMINTEG_FILE.format(sim.name, sim.dt, sim.steps)

    # Calculate with caching
    integ = calculate_integrals(sim.mm, mu_arr, integ_cache)
    int_mm = SimintegMemmap(integ_cache)

    # Plot
    h_0 = int_mm.h[0]
    t_ = sim.mm.t / 3600 / 24  # convert to days

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t_, (int_mm.h[:] - h_0) / h_0 * 100, label="Specific angular momentum")
    plt.ylabel("Change [%]")
    plt.title("Change in Angular Momentum")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(
        t_,
        (int_mm.e - int_mm.e[0]) / int_mm.e[0] * 100,
        label="Total Energy",
        linewidth=2,
    )
    plt.ylabel("Change [%]")
    plt.xlabel("Time [days]")
    plt.title("Change in Energy Components")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Initial total energy: {int_mm.e[0]:.6e}")
    print(f"Final total energy: {int_mm.e[-1]:.6e}")
    print(
        f"Energy conservation: {((int_mm.e[-1] - int_mm.e[0]) / int_mm.e[0] * 100):.6f}%"
    )
