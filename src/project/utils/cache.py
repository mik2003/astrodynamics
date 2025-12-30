import struct
from io import BufferedReader, BufferedWriter
from pathlib import Path
from types import EllipsisType
from typing import Literal, Sequence, Tuple, Union

import numpy as np

from project.utils import FloatArray, IntArray

SIMSTATE_EXTENSION = ".simstate"
SIMSTATE_FILE = "{}__{}__{}" + SIMSTATE_EXTENSION  # name, dt, steps

MAGIC = b"SIMSTATE"
VERSION = 1
HEADER_FMT = (
    "<"  # little-endian
    "8s"  # magic
    "I"  # version
    "Q"  # steps
    "I"  # bodies
    "I"  # state_dim
    "d"  # dt
    "28s"  # future / padding
)
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 64 bytes


def write_header(
    f: BufferedWriter,
    steps: int,
    bodies: int,
    state_dim: int,
    dt: float,
) -> None:
    f.write(
        struct.pack(
            HEADER_FMT,
            MAGIC,
            VERSION,
            steps,
            bodies,
            state_dim,
            dt,
            b"\x00" * 28,
        )
    )


def read_header(f: BufferedReader) -> Tuple[int, int, int, float]:
    magic, version, steps, bodies, state_dim, dt, _ = struct.unpack(
        HEADER_FMT, f.read(64)
    )

    if magic != MAGIC:
        raise ValueError("Not a SIMSTATE file")

    if version != VERSION:
        raise ValueError(f"File version {version} != expected {VERSION}")

    return steps, bodies, state_dim, dt


def write_simstate(filename: Path, data: FloatArray) -> None:
    """
    Write a complete simulation file (.simstate) from a full array.

    Parameters
    ----------
    filename : Path
        Path to the output file.
    data : np.ndarray
        Shape (steps, bodies, state_dim), dtype float64.
    """
    parse_simstate_filename(filename=filename)

    if data.ndim != 3:
        raise ValueError("data must be (steps, bodies, state_dim)")

    data = np.asarray(data, dtype=np.float64)

    steps, bodies, state_dim, dt = validate_simstate_data(filename=filename, data=data)

    with open(filename, "wb") as f:
        write_header(f, steps, bodies, state_dim, dt)
        data.astype(np.float64, copy=False).tofile(f)


def read_simstate(filename: Path) -> Tuple[np.memmap, Tuple[int, int, int]]:
    """
    Read a .simstate file into memory.

    Returns
    -------
    data : np.ndarray
        Array of shape (steps, bodies, state_dim)
    header : tuple
        (steps, bodies, state_dim)
    """
    parse_simstate_filename(filename=filename)

    with open(filename, "rb") as f:
        steps, bodies, state_dim = validate_simstate_file(filename=filename, file=f)
        # Memmap of remaining data
        mm = np.memmap(
            filename=filename,
            dtype="float64",
            mode="r",
            offset=HEADER_SIZE,
            shape=(steps, bodies, state_dim),
        )

    return mm, (steps, bodies, state_dim)


def parse_simstate_filename(filename: Path) -> Tuple[str, int, int]:
    if filename.suffix != SIMSTATE_EXTENSION:
        raise ValueError(f"{filename} is not a .simstate binary file")
    elements = filename.stem.split("__")
    if len(elements) != 3:
        raise ValueError(
            f"'{filename.name}' Filename format is wrong (name__dt__steps.simstate)"
        )
    name, dt, steps = elements

    return name, int(dt), int(steps)


def validate_simstate_file(
    filename: Path, file: BufferedReader
) -> Tuple[int, int, int]:
    _, dt_f, steps_f = parse_simstate_filename(filename=filename)
    steps, bodies, state_dim, dt = read_header(f=file)
    if dt_f != dt or steps_f != steps - 1:
        raise ValueError("Filename does not match file header")

    return steps, bodies, state_dim


def validate_simstate_data(
    filename: Path, data: FloatArray
) -> Tuple[int, int, int, int]:
    _, dt_f, steps_f = parse_simstate_filename(filename=filename)
    steps, bodies, state_dim = data.shape
    if steps_f != steps - 1:
        raise ValueError(
            f"{steps - 1} actual simulation steps do not match filename's {steps_f}"
        )

    return steps, bodies, state_dim, dt_f


def simstate_view_from_state_view(y: FloatArray, n: int) -> FloatArray:
    # Interleave positions and velocities per body
    steps, _ = y.shape
    y_r = y[:, : 3 * n].reshape(steps, n, 3)
    y_v = y[:, 3 * n :].reshape(steps, n, 3)
    return np.concatenate([y_r, y_v], axis=-1)  # shape = (steps, n, 6)


Index = Union[
    int,
    slice,
    Sequence[int],
    IntArray,
    EllipsisType,
    None,
]


class Memmap:
    def __init__(self, filename: Path) -> None:
        mm, (steps, bodies, state_dim) = read_simstate(filename)
        self.mm = mm
        self.steps = steps
        self.bodies = bodies

        self._r_view = _RVView(self, "r")
        self._v_view = _RVView(self, "v")

    @property
    def r(self) -> "_RVView":
        return self._r_view

    @property
    def v(self) -> "_RVView":
        return self._v_view

    def __getitem__(self, key: Index | Tuple[Index, ...]) -> FloatArray:
        return self.mm[key]


class _RVView:
    def __init__(self, parent: "Memmap", rv: Literal["r", "v"]) -> None:
        self._parent = parent
        self._rv = rv

    def __getitem__(self, key: Index | Tuple[Index, ...]) -> FloatArray:
        """
        NumPy-like indexing:
            mm.r[step, body]
            mm.r[step_slice, body_slice]
            mm.r[..., :]
        """
        data = self._parent.mm
        # Determine slice for last dimension
        if self._rv == "r":
            last_dim = slice(0, 3)
        elif self._rv == "v":
            last_dim = slice(3, 6)
        else:
            raise ValueError("rv must be 'r' or 'v'")

        # If user provided full indexing, append last dimension
        if isinstance(key, tuple):
            # Append last dim slice if key has 2 dims or fewer
            if len(key) == 1:
                key = key + (last_dim,)
            elif len(key) == 2:
                key = key + (last_dim,)
            # else, assume user already indexed last dim
        else:
            # Single int or slice → apply to first dim
            key = (key, slice(None), last_dim)

        return data[key]
