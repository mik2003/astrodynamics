# integrals.py
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from project.simulation import Simulation
from project.utils import A, Dir, print_done, print_progress

G = 6.67430e-11


def calculate_energy(mm: np.memmap, mu: A, cache_file: Path | None = None) -> A:
    """
    Calculate total energy using vectorized operations with caching
    """
    # Check cache first
    if cache_file and cache_file.exists():
        print("Loading energy from cache...")
        mm = np.memmap(cache_file, dtype="float64", mode="r", shape=(mm.shape[0],))
        return np.array(mm)

    n_steps, _, n_bodies = mm.shape

    # Kinetic energy (fast)
    v_sq = np.sum(mm[:, 3:6, :] ** 2, axis=1)
    masses = mu / G
    e_kin = 0.5 * np.sum(v_sq * masses, axis=1)

    # Potential energy (slow)
    e_pot = np.zeros(n_steps)
    i_idx, j_idx = np.triu_indices(n_bodies, k=1)

    start_time = time.time()
    print("Calculating energy...")

    for t in range(n_steps):
        if t % 10000 == 0:
            print_progress(t, n_steps, start_time)

        r_i = mm[t, 0:3, i_idx].T
        r_j = mm[t, 0:3, j_idx].T
        r_diff = r_i - r_j
        distances = np.linalg.norm(r_diff, axis=0)
        mu_pairs = mu[i_idx] * mu[j_idx]

        with np.errstate(divide="ignore", invalid="ignore"):
            potential_terms = mu_pairs / (G * distances)
            potential_terms = np.nan_to_num(
                potential_terms, nan=0.0, posinf=0.0, neginf=0.0
            )

        e_pot[t] = -np.sum(potential_terms)

    print_done()
    print("Energy calculation complete.")

    e_total = e_kin + e_pot

    # Save to cache
    if cache_file:
        print(f"Saving energy to cache: {cache_file}")
        mm = np.memmap(cache_file, dtype="float64", mode="w+", shape=e_total.shape)
        mm[:] = e_total[:]
        mm.flush()

    return e_total


def calculate_angular_momentum(
    mm: np.memmap, mu: A, cache_file: Path | None = None
) -> A:
    """
    Calculate angular momentum with caching
    """
    if cache_file and cache_file.exists():
        print("Loading angular momentum from cache...")
        mm_ = np.memmap(cache_file, dtype="float64", mode="r", shape=(mm.shape[0], 3))
        return np.array(mm_)

    masses_ = mu / G
    v_weighted = mm[:, 3:6, :] * masses_[None, None, :]
    h = np.sum(np.cross(mm[:, 0:3, :], v_weighted, axis=1), axis=2)

    # Save to cache
    if cache_file:
        print(f"Saving angular momentum to cache: {cache_file}")
        mm_ = np.memmap(cache_file, dtype="float64", mode="w+", shape=h.shape)
        mm_[:] = h[:]
        mm_.flush()

    return h


if __name__ == "__main__":
    sim = Simulation(
        name="solar_system_moons_2460966",
        dt=3600,
        time=3600 * 24 * 365.25 * 100,
    )

    mu_list = [body.mu for body in sim.body_list]
    mu_arr = np.array(mu_list)

    # Create cache filenames based on simulation parameters
    base_name = f"{sim.name}_{sim.dt}_{sim.steps}"
    energy_cache = Dir.data.joinpath(f"{base_name}_energy.bin")
    angular_cache = Dir.data.joinpath(f"{base_name}_angular.bin")

    # Calculate with caching
    e_total = calculate_energy(sim.mm, mu_arr, energy_cache)
    h = calculate_angular_momentum(sim.mm, mu_arr, angular_cache)

    # Rest of plotting code remains the same...
    h_0 = h[0, 2]

    # Plotting
    t_ = np.arange(0, sim.time, sim.dt) / 3600 / 24

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t_, (h[:, 2] - h_0) / h_0 * 100, label="Specific angular momentum")
    plt.ylabel("Change [%]")
    plt.title("Change in Angular Momentum")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(
        t_,
        (e_total - e_total[0]) / e_total[0] * 100,
        label="Total Energy",
        linewidth=2,
    )
    plt.ylabel("Change [%]")
    plt.xlabel("Time [day]")
    plt.title("Change in Energy Components")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Initial total energy: {e_total[0]:.6e}")
    print(f"Final total energy: {e_total[-1]:.6e}")
    print(
        "Energy conservation: "
        + f"{((e_total[-1] - e_total[0]) / e_total[0] * 100):.6f}%"
    )
