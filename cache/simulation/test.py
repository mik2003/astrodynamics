from pathlib import Path

import numpy as np

from project.utils import Dir
from project.utils.cache import Memmap

fname = "sun_earth_moon_2460966"
dt = "3600"
steps = "8766"

f_bin: Path = Dir.simulation / "{}_{}_{}.bin".format(fname, dt, steps)
f_sim: Path = Dir.simulation / "{}__{}__{}.simstate".format(fname, dt, steps)

data_bin = np.memmap(
    filename=f_bin,
    dtype="float64",
    mode="r",
    shape=(int(steps), 9, 3),
)

data_sim = Memmap(f_sim)

print(data_bin[0, :3, :])
print(data_sim.r_vis[0, :])

np.testing.assert_allclose(
    data_bin[-1, :3, :][np.newaxis, :, :], data_sim.r_vis[8765, :]
)
