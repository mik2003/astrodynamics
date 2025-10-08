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

r = mm[:, 0:3, :]
v = mm[:, 3:6, :]
a = mm[:, 6:9, :]

h = np.sum(np.cross(r, v, axis=1), axis=2)
h_0 = h[0, 2]

v_square = np.sum(np.square(v), axis=1)
e_k = np.sum(v_square / 2, axis=1)
mu = 0
for body in body_list:
    mu += body.mu
e_p = -mu / np.sqrt(np.sum(np.square(np.sum(r, axis=1)), axis=1))
e = e_k + e_p

t = np.arange(0, time, dt) / 3600 / 24
plt.plot(t, (h[:, 2] - h_0) / h_0 * 100, label="Specific angular momentum h")
plt.plot(t, (e - e[0]) / e[0] * 100, label="Orbital energy E")
plt.title("Change in the integrals of orbital motion")
plt.xlabel("Time [day]")
plt.ylabel("Change [%]")
plt.legend()
plt.show()
