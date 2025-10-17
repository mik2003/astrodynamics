import time

import matplotlib.pyplot as plt
import numpy as np

from project.simulation import Simulation

G = 6.67430e-11


def calculate_energy(r, v, mu):
    """
    Calculate total energy using vectorized operations
    mu: array of gravitational parameters [m^3/s^2]
    """
    n_steps, _, n_bodies = r.shape

    # Calculate kinetic energy: E_kin = 0.5 * sum((mu_i/G) * v_i^2)
    v_sq = np.sum(v**2, axis=1)  # shape: (n_steps, n_bodies)
    masses = mu / G  # shape: (n_bodies,)
    e_kin = 0.5 * np.sum(v_sq * masses, axis=1)  # shape: (n_steps,)

    # Calculate potential energy using vectorized pairwise distances
    e_pot = np.zeros(n_steps)

    # Create indices for all unique pairs
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

        # Extract positions and transpose to get (3, n_pairs) shape
        r_i = r[
            t, :, i_idx
        ].T  # shape: (n_pairs, 3) -> transpose to (3, n_pairs)
        r_j = r[
            t, :, j_idx
        ].T  # shape: (n_pairs, 3) -> transpose to (3, n_pairs)

        r_diff = r_i - r_j  # shape: (3, n_pairs)

        # Calculate distances (norm along axis 0)
        distances = np.linalg.norm(r_diff, axis=0)  # shape: (n_pairs,)

        # Calculate potential energy terms
        mu_pairs = mu[i_idx] * mu[j_idx]  # shape: (n_pairs,)

        # Calculate potential energy (avoid division by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            potential_terms = mu_pairs / (G * distances)
            potential_terms = np.nan_to_num(
                potential_terms, nan=0.0, posinf=0.0, neginf=0.0
            )

        e_pot[t] = -np.sum(potential_terms)

    print(
        "\rProgress [" + "#" * 50 + "] 100.00% | ETA: 00:00:00"
    )  # Show full bar at the end
    print("Energy calculation complete.")

    return e_kin + e_pot


if __name__ == "__main__":

    sim = Simulation(
        name="inner_solar_system_2460959",
        dt=3600,  # simulation time step (seconds)
        time=3600 * 24 * 365.25 * 100,  # simulation time (seconds)
    )

    # Extract gravitational parameters
    mu_list = [body.mu for body in sim.body_list]
    mu_arr = np.array(mu_list)

    # Calculate energy with progress bar
    e_total = calculate_energy(sim.r, sim.v, mu_arr)

    # Optimized angular momentum calculation
    # r shape: (steps, 3, num_bodies), v shape: (steps, 3, num_bodies)
    # We need to multiply v by mass (mu/G) for each body
    masses_ = mu_arr / G  # shape: (num_bodies,)
    v_weighted = (
        sim.v * masses_[None, None, :]
    )  # Broadcast masses across time and coordinates

    # Calculate angular momentum: h = sum(r × (m*v))
    # for all bodies at each time step
    h = np.sum(
        np.cross(sim.r, v_weighted, axis=1), axis=2
    )  # shape: (steps, 3)
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
