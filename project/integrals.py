import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from project.data import BodyList
from project.formulas import simulate_n_steps
from project.utilities import A, Dir, load_trails_npy

# Simulation parameters
dt = 3600  # simulation time step (seconds)
time = 3600 * 24 * 365.25  # simulation time (seconds)
steps = int(time / dt)  # total number of simulation steps

fname = "test"
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

r = mm[:, 0:3, :]
v = mm[:, 3:6, :]
a = mm[:, 6:9, :]

h = np.sum(np.cross(r, v, axis=1), axis=2)
h_0 = h[0, 2]
print(h)

mu_list = []
for body in body_list:
    mu_list.append(body.mu)
mu = np.array(mu_list)
v_square = np.sum(np.square(v), axis=1)
e_k = np.sum(v_square / 2 * mu.T, axis=1) / np.sum(mu)
e_p = np.sum(
    -np.sum(mu**2) / np.sqrt(np.sum(np.square(r[:, :, 1:]), axis=1)), axis=1
) / np.sum(mu)
print(e_p)
e = e_k + e_p

t = np.arange(0, time, dt) / 3600 / 24
plt.plot(t, (h[:, 2] - h_0) / h_0 * 100, label="Specific angular momentum h")
plt.plot(t, (e - e[0]) / e[0] * 100, label="Orbital energy E")
plt.plot(t, (e_k - e_k[0]) / e_k[0] * 100, label="E_k")
plt.plot(t, (e_p - e_p[0]) / e_p[0] * 100, label="E_p")
plt.ylabel("Change [%]")
# plt.plot(t, h[:, 2], label="Specific angular momentum h [m^2 s^-1]")
# plt.plot(t, e_k, label="E_k [m^2 s^-2]")
# plt.plot(t, e_p, label="E_p [m^2 s^-2]")
# plt.plot(t, e, label="Orbital energy E")
# plt.ylabel("Value")
plt.title("Change in the integrals of orbital motion")
plt.xlabel("Time [day]")
plt.legend()
plt.show()
