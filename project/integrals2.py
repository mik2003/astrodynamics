import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from project.data import BodyList
from project.formulas import simulate_n_steps
from project.utilities import A, Dir, load_trails_npy

# Simulation parameters
dt = 3600  # simulation time step (seconds)
time = 3600 * 24 * 365.25 * 100  # simulation time (seconds)
steps = int(time / dt)  # total number of simulation steps

fname = "sun_earth"
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

# Extract masses or gravitational parameters
# Assuming body.mu is GM (gravitational parameter)
mu_list = []
mass_list = []
for body in body_list:
    mu_list.append(body.mu)
    # If you have actual mass, use it instead:
    # mass_list.append(body.mass)
mu = np.array(mu_list)

# If mu is GM, then mass = mu / G, but for energy calculations we can work with mu
# Let's assume mu contains GM values


# If you want to work entirely with gravitational parameters:
def calculate_energy_simple(r, v, mu):
    """
    Calculate energy using gravitational parameters directly
    mu: array of gravitational parameters [m^3/s^2]
    """
    n_steps, _, n_bodies = r.shape
    G = 6.67430e-11

    e_total = np.zeros(n_steps)

    for t in range(n_steps):
        # Kinetic energy: E_kin = 0.5 * sum((mu_i/G) * v_i^2)
        e_kin = 0.0
        for i in range(n_bodies):
            v_sq = np.sum(v[t, :, i] ** 2)
            e_kin += 0.5 * (mu[i] / G) * v_sq

        # Potential energy: E_pot = -sum_{i<j} (mu_i * mu_j) / (G * r_ij)
        e_pot = 0.0
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                r_ij = r[t, :, i] - r[t, :, j]
                distance = np.sqrt(np.sum(r_ij**2))
                if distance > 0:
                    e_pot -= (mu[i] * mu[j]) / (G * distance)

        e_total[t] = e_kin + e_pot

    return e_total


# Usage:
e_total = calculate_energy_simple(r, v, mu)

# Angular momentum calculation (your original seems correct)
h = np.sum(np.cross(r, v * mu[np.newaxis, :], axis=1), axis=2)  # Include mass
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
