import os

import matplotlib.pyplot as plt
import numpy as np

from project.data import BodyList
from project.formulas import simulate_n_steps
from project.utilities import Dir

# Simulation parameters
dt = 3600  # simulation time step (seconds)
time = 3600 * 24 * 365.25  # simulation time (seconds)
steps = int(time / dt)  # total number of simulation steps

fname = "full_solar_system_with_dwarf_planets"
file_in = Dir.data_dir.joinpath(fname + ".json")
file_traj = Dir.data_dir.joinpath(f"{fname}_{dt}_{steps}.bin")

# Run simulation first and save trajectory with progress tracker
body_list = BodyList.load(file_in)
if not os.path.exists(file_traj):
    print(f"Simulating {time:.2e} seconds...")
    simulate_n_steps(body_list, steps, dt, file_traj, prnt=True)
    print("\nSimulation complete.")

num_bodies = len(body_list)

mm = np.memmap(
    file_traj,
    dtype="float64",
    mode="r",
    shape=(steps, 9, num_bodies),
)

r = mm[:, 0:3, :]  # positions
v = mm[:, 3:6, :]  # velocities

# Extract gravitational parameters
mu_list = [body.mu for body in body_list]
mu = np.array(mu_list)
G = 6.67430e-11


def calculate_energy_vectorized(r, v, mu):
    """
    Calculate energy using vectorized operations
    mu: array of gravitational parameters [m^3/s^2]
    """
    n_steps, _, n_bodies = r.shape

    # Calculate kinetic energy: E_kin = 0.5 * sum((mu_i/G) * v_i^2)
    # Vectorized over all bodies and time steps
    v_sq = np.sum(v**2, axis=1)  # shape: (n_steps, n_bodies)
    masses = mu / G  # shape: (n_bodies,)
    e_kin = 0.5 * np.sum(v_sq * masses, axis=1)  # shape: (n_steps,)

    # Calculate potential energy using vectorized pairwise distances
    e_pot = np.zeros(n_steps)

    # Create indices for all unique pairs
    i_idx, j_idx = np.triu_indices(n_bodies, k=1)

    for t in range(n_steps):
        # Vectorized distance calculation for all pairs at time t
        r_diff = r[t, :, i_idx] - r[t, :, j_idx]  # shape: (3, n_pairs)
        distances = np.sqrt(np.sum(r_diff**2, axis=0))  # shape: (n_pairs,)

        # Vectorized potential energy calculation for all pairs
        mu_pairs = mu[i_idx] * mu[j_idx]  # shape: (n_pairs,)
        e_pot[t] = -np.sum(mu_pairs / (G * distances))

    return e_kin + e_pot


def calculate_energy_fully_vectorized(r, v, mu):
    """
    Fully vectorized version (memory intensive for large n_bodies)
    """
    n_steps, _, n_bodies = r.shape

    # Kinetic energy (same as above)
    v_sq = np.sum(v**2, axis=1)
    masses = mu / G
    e_kin = 0.5 * np.sum(v_sq * masses, axis=1)

    # Potential energy - fully vectorized across time
    i_idx, j_idx = np.triu_indices(n_bodies, k=1)
    n_pairs = len(i_idx)

    # Calculate all pairwise differences for all time steps
    # This creates a large array: (n_steps, 3, n_pairs)
    r_diff = r[:, :, i_idx] - r[:, :, j_idx]

    # Calculate distances for all pairs and all time steps
    distances = np.sqrt(np.sum(r_diff**2, axis=1))  # shape: (n_steps, n_pairs)

    # Calculate potential energy for all pairs
    mu_pairs = mu[i_idx] * mu[j_idx]  # shape: (n_pairs,)
    e_pot = -np.sum(
        mu_pairs[None, :] / (G * distances), axis=1
    )  # shape: (n_steps,)

    return e_kin + e_pot


# Use the appropriate version based on your memory constraints
if num_bodies <= 10:  # Use fully vectorized for small number of bodies
    e_total = calculate_energy_fully_vectorized(r, v, mu)
else:  # Use semi-vectorized for larger number of bodies
    e_total = calculate_energy_vectorized(r, v, mu)

# Optimized angular momentum calculation
# r shape: (steps, 3, num_bodies), v shape: (steps, 3, num_bodies)
# We need to multiply v by mass (mu/G) for each body
masses = mu / G  # shape: (num_bodies,)
v_weighted = (
    v * masses[None, None, :]
)  # Broadcast masses across time and coordinates

# Calculate angular momentum: h = sum(r × (m*v)) for all bodies at each time step
h = np.sum(np.cross(r, v_weighted, axis=1), axis=2)  # shape: (steps, 3)
h_0 = h[0, 2]

# Plotting
t = np.arange(0, time, dt) / 3600 / 24

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, (h[:, 2] - h_0) / h_0 * 100, label="Specific angular momentum")
plt.ylabel("Change [%]")
plt.title("Change in Angular Momentum")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(
    t,
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
    f"Energy conservation: {((e_total[-1] - e_total[0]) / e_total[0] * 100):.6f}%"
)
