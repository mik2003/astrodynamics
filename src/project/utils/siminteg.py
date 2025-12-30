import struct
from io import BufferedReader, BufferedWriter
from pathlib import Path
from typing import Tuple, cast

import numpy as np
import scipy

from project.utils import FloatArray

SIMINTEG_EXTENSION = ".siminteg"
SIMINTEG_FILE = "{}__{}__{}" + SIMINTEG_EXTENSION

MAGIC = b"SIMINTEG"
VERSION = 2
HEADER_FMT = (
    "<"  # little-endian
    "8s"  # magic
    "I"  # version
    "Q"  # steps
    "I"  # integ_dim
    "d"  # dt
    "32s"  # future / padding
)
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 64 bytes


def write_header(
    f: BufferedWriter,
    steps: int,
    integ_dim: int,
    dt: float,
) -> None:
    f.write(
        struct.pack(
            HEADER_FMT,
            MAGIC,
            VERSION,
            steps,
            integ_dim,
            dt,
            b"\x00" * 32,
        )
    )


def read_header(f: BufferedReader) -> Tuple[int, int, float]:
    magic, version, steps, integ_dim, dt, _ = struct.unpack(HEADER_FMT, f.read(64))

    if magic != MAGIC:
        raise ValueError("Not a SIMSTATE file")

    if version != VERSION:
        raise ValueError(f"File version {version} != expected {VERSION}")

    return steps, integ_dim, dt


def write_siminteg(filename: Path, data: FloatArray) -> None:
    """
    Write a complete simulation integrals file (.siminteg) from an array.

    Parameters
    ----------
    filename : Path
        Path to the output file.
    data : np.ndarray
        Shape (steps, integ_dim), dtype float64.
    """
    if data.ndim != 2:
        raise ValueError("data must be (steps, integ_dim)")

    data = np.asarray(data, dtype=np.float64)

    steps, integ_dim, dt = validate_siminteg_data(filename=filename, data=data)

    with open(filename, "wb") as f:
        write_header(f, steps, integ_dim, dt)
        data.astype(np.float64, copy=False).tofile(f)


def read_siminteg(
    filename: Path,
) -> Tuple[np.memmap, Tuple[int, int, float]]:
    """
    Read a .siminteg file into memory.

    Returns
    -------
    data : np.ndarray
        Array of shape (integ_dim, steps)
    header : tuple
        (steps, integ_dim, dt)
    """
    with open(filename, "rb") as f:
        steps, integ_dim, dt = validate_siminteg_file(filename=filename, file=f)

    # Memmap of remaining data
    mm = np.memmap(
        filename=filename,
        dtype="float64",
        mode="r",
        offset=HEADER_SIZE,
        shape=(steps, integ_dim),
    )

    return mm, (steps, integ_dim, dt)


def parse_siminteg_filename(filename: Path) -> Tuple[str, int, int]:
    if filename.suffix != SIMINTEG_EXTENSION:
        raise ValueError(f"{filename} is not a .siminteg binary file")
    elements = filename.stem.split("__")
    if len(elements) != 3:
        raise ValueError(
            f"'{filename.name}' Filename format is wrong (name__dt__steps.siminteg)"
        )
    name, dt, steps = elements

    return name, int(dt), int(steps)


def validate_siminteg_file(
    filename: Path, file: BufferedReader
) -> Tuple[int, int, float]:
    _, dt_f, steps_f = parse_siminteg_filename(filename=filename)
    steps, integ_dim, dt = read_header(f=file)
    if dt_f != int(dt) or steps_f != steps - 1:
        raise ValueError("Filename does not match file header")

    return steps, integ_dim, dt


def validate_siminteg_data(filename: Path, data: FloatArray) -> Tuple[int, int, int]:
    _, dt_f, steps_f = parse_siminteg_filename(filename=filename)
    steps, integ_dim = data.shape
    if steps_f != steps - 1:
        raise ValueError(
            f"{steps - 1} actual simulation steps do not match filename's {steps_f}"
        )
    return steps, integ_dim, dt_f


class SimintegMemmap:
    def __init__(self, filename: Path) -> None:
        mm, (steps, integ_dim, dt) = read_siminteg(filename)
        self.mm = mm
        self.steps = steps
        self.integ_dim = integ_dim
        self.dt = dt

    @property
    def e(self) -> FloatArray:
        return self.mm[:, 0]

    @property
    def h_vec(self) -> FloatArray:
        return self.mm[:, 1:]

    @property
    def h(self) -> FloatArray:
        return cast(FloatArray, scipy.linalg.norm(self.h_vec, axis=1))
