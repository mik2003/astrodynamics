import os
import pathlib
from typing import List

import numpy as np
import numpy.typing as npt

A = npt.NDArray[np.float64]


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case"""
    import re

    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


class Dir:
    """Directories."""

    main_dir = pathlib.Path(__file__).parent.absolute()
    data_dir = pathlib.Path(main_dir).parent.absolute().joinpath("data")


def append_positions_npy(
    filename: pathlib.Path, positions_buffer: np.ndarray
) -> None:
    """
    Append positions to a binary file.
    positions_buffer: shape (buffer_size, num_bodies, 3)
    """
    with open(filename, "ab") as f:
        positions_buffer.astype(np.float64).tofile(f)


def load_trails_npy(
    filename: str, last_n_steps: int, step_interval: int, num_bodies: int
) -> List[A]:
    """
    Load the last_n_steps (3, num_bodies) arrays from a binary file.
    Returns a list of np.ndarray of shape (3, num_bodies) for each step.
    """
    record_size = 3 * num_bodies
    file_size = os.path.getsize(filename) // 8  # float64 bytes
    num_records = file_size // record_size
    mm = np.memmap(
        filename, dtype="float64", mode="r", shape=(num_records, 3, num_bodies)
    )
    indices = np.arange(num_records - 1, -1, -step_interval)[:last_n_steps]
    trails = [mm[i] for i in indices[::-1]]  # chronological order
    return trails
