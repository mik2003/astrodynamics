import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from project.data import BodyList
from project.formulas import simulate_n_steps
from project.utilities import A, Dir, load_trails_npy

# Simulation parameters
dt = 3600  # simulation time step (seconds)
time = 3600 * 24 * 365.25 * 10  # simulation time (seconds)
steps = int(time / dt)  # total number of simulation steps

fname = "sun_earth"
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

print(mm.shape)

r = mm[:, 0:3, :]
v = mm[:, 3:6, :]
a = mm[:, 6:9, :]

h = np.sum(np.cross(r, v, axis=1), axis=2)
h_0 = h[0, 2]

plt.plot(np.arange(0, time, dt) / 3600 / 24, (h[:, 2] - h_0) / h_0 * 100)
plt.title("Percentage Change in Specific Angular Momentum h")
plt.xlabel("Time [day]")
plt.ylabel("Change [%]")
plt.show()
