# integrals.py
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from project.simulation import Simulation
from project.utilities import Dir

G = 6.67430e-11


def calculate_energy(r, v, mu, cache_file: Path | None = None):
    """
    Calculate total energy using vectorized operations with caching
    """
    # Check cache first
    if cache_file and cache_file.exists():
        print("Loading energy from cache...")
        mm = np.memmap(
            cache_file, dtype="float64", mode="r", shape=(r.shape[0],)
        )
        return np.array(mm)

    n_steps, _, n_bodies = r.shape

    # Kinetic energy (fast)
    v_sq = np.sum(v**2, axis=1)
    masses = mu / G
    e_kin = 0.5 * np.sum(v_sq * masses, axis=1)

    # Potential energy (slow)
    e_pot = np.zeros(n_steps)
    i_idx, j_idx = np.triu_indices(n_bodies, k=1)

    start_time = time.time()
    print("Calculating energy...")

    for t in range(n_steps):
        if t % 100 == 0:
            progress = int(t / n_steps * 50)
            bar = "[" + "#" * progress + "-" * (50 - progress) + "]"
            elapsed = time.time() - start_time
            if t > 0:
                est_total = elapsed / (t / n_steps)
                est_remain = est_total - elapsed
                hrs = int(est_remain // 3600)
                mins = int((est_remain % 3600) // 60)
                secs = int(est_remain % 60)
                est_str = f" | ETA: {hrs:02d}:{mins:02d}:{secs:02d}"
            else:
                est_str = ""
            print(f"\rProgress {bar} {t/n_steps*100:.2f}%{est_str}", end="")

        r_i = r[t, :, i_idx].T
        r_j = r[t, :, j_idx].T
        r_diff = r_i - r_j
        distances = np.linalg.norm(r_diff, axis=0)
        mu_pairs = mu[i_idx] * mu[j_idx]

        with np.errstate(divide="ignore", invalid="ignore"):
            potential_terms = mu_pairs / (G * distances)
            potential_terms = np.nan_to_num(
                potential_terms, nan=0.0, posinf=0.0, neginf=0.0
            )

        e_pot[t] = -np.sum(potential_terms)

    print("\rProgress [" + "#" * 50 + "] 100.00% | ETA: 00:00:00")
    print("Energy calculation complete.")

    e_total = e_kin + e_pot

    # Save to cache
    if cache_file:
        print(f"Saving energy to cache: {cache_file}")
        mm = np.memmap(
            cache_file, dtype="float64", mode="w+", shape=e_total.shape
        )
        mm[:] = e_total[:]
        mm.flush()

    return e_total


def calculate_angular_momentum(r, v, mu, cache_file: Path | None = None):
    """
    Calculate angular momentum with caching
    """
    if cache_file and cache_file.exists():
        print("Loading angular momentum from cache...")
        mm = np.memmap(
            cache_file, dtype="float64", mode="r", shape=(r.shape[0], 3)
        )
        return np.array(mm)

    masses_ = mu / G
    v_weighted = sim.v * masses_[None, None, :]
    h = np.sum(np.cross(sim.r, v_weighted, axis=1), axis=2)

    # Save to cache
    if cache_file:
        print(f"Saving angular momentum to cache: {cache_file}")
        mm = np.memmap(cache_file, dtype="float64", mode="w+", shape=h.shape)
        mm[:] = h[:]
        mm.flush()

    return h


if __name__ == "__main__":
    sim = Simulation(
        name="sun_earth_moon_2460966",
        dt=3600,
        time=3600 * 24 * 365.25 * 100,
    )

    mu_list = [body.mu for body in sim.body_list]
    mu_arr = np.array(mu_list)

    # Create cache filenames based on simulation parameters
    base_name = f"{sim.name}_{sim.dt}_{sim.steps}"
    energy_cache = Dir.data_dir.joinpath(f"{base_name}_energy.bin")
    angular_cache = Dir.data_dir.joinpath(f"{base_name}_angular.bin")

    # Calculate with caching
    e_total = calculate_energy(sim.r, sim.v, mu_arr, energy_cache)
    h = calculate_angular_momentum(sim.r, sim.v, mu_arr, angular_cache)

    # Rest of plotting code remains the same...
    h_0 = h[0, 2]

    # Plotting
    t_ = np.arange(0, sim.time, sim.dt) / 3600 / 24

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(
        t_, (h[:, 2] - h_0) / h_0 * 100, label="Specific angular momentum"
    )
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
