import struct
from io import BufferedReader, BufferedWriter
from pathlib import Path
from typing import Literal, Tuple

import numpy as np

from project.utils import FloatArray, Index

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


def write_simstate(
    filename: Path, data: FloatArray, t: FloatArray | None = None
) -> None:
    """
    Write a complete simulation file (.simstate) from a full array.

    Parameters
    ----------
    filename : Path
        Path to the output file.
    data : np.ndarray
        Shape (steps, bodies, state_dim), dtype float64.
    """
    if data.ndim != 3:
        raise ValueError("data must be (steps, bodies, state_dim)")

    data = np.asarray(data, dtype=np.float64)

    steps, bodies, state_dim, dt = validate_simstate_data(
        filename=filename, data=data, t=t
    )

    with open(filename, "wb") as f:
        write_header(f, steps, bodies, state_dim, dt)
        data.astype(np.float64, copy=False).tofile(f)


def read_simstate(
    filename: Path,
) -> Tuple[np.memmap, Tuple[int, int, int, float], np.memmap | None]:
    """
    Read a .simstate file into memory.

    Returns
    -------
    data : np.ndarray
        Array of shape (steps, bodies, state_dim)
    header : tuple
        (steps, bodies, state_dim, dt)
    t : np.ndarray | None
        Array of shape (steps,)
    """
    with open(filename, "rb") as f:
        steps, bodies, state_dim, dt = validate_simstate_file(filename=filename, file=f)

    # Memmap of remaining data
    mm = np.memmap(
        filename=filename,
        dtype="float64",
        mode="r",
        offset=HEADER_SIZE,
        shape=(steps, bodies, state_dim),
    )

    if dt < 0:
        t = np.memmap(
            filename=filename,
            dtype="float64",
            mode="r",
            offset=HEADER_SIZE + mm.nbytes,
            shape=(steps,),
        )
    else:
        t = None

    return mm, (steps, bodies, state_dim, dt), t


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
) -> Tuple[int, int, int, float]:
    _, dt_f, steps_f = parse_simstate_filename(filename=filename)
    steps, bodies, state_dim, dt = read_header(f=file)
    if dt_f != int(dt) or steps_f != steps - 1:
        raise ValueError("Filename does not match file header")

    return steps, bodies, state_dim, dt


def validate_simstate_data(
    filename: Path, data: FloatArray, t: FloatArray | None = None
) -> Tuple[int, int, int, int]:
    _, dt_f, steps_f = parse_simstate_filename(filename=filename)
    steps, bodies, state_dim = data.shape
    if steps_f != steps - 1:
        raise ValueError(
            f"{steps - 1} actual simulation steps do not match filename's {steps_f}"
        )
    if t is not None and t.size != steps:
        raise ValueError("Time vector size is inconsistent with state matrix")

    return steps, bodies, state_dim, dt_f


def simstate_view_from_state_view(y: FloatArray, n: int) -> FloatArray:
    # Interleave positions and velocities per body
    steps, _ = y.shape
    y_r = y[:, : 3 * n].reshape(steps, n, 3)
    y_v = y[:, 3 * n :].reshape(steps, n, 3)
    return np.concatenate([y_r, y_v], axis=-1)  # shape = (steps, n, 6)


class SimstateMemmap:
    def __init__(self, filename: Path) -> None:
        mm, (steps, bodies, state_dim, dt), t = read_simstate(filename)
        self.mm = mm
        self.steps = steps
        self.bodies = bodies
        self.state_dim = state_dim
        self.dt = dt

        self._t = t

        self._r_view = _RVView(self, "r")
        self._v_view = _RVView(self, "v")

        self._r_vis_view = _RVVisView(self._r_view)
        self._v_vis_view = _RVVisView(self._v_view)

    @property
    def r(self) -> "_RVView":
        return self._r_view

    @property
    def v(self) -> "_RVView":
        return self._v_view

    @property
    def r_vis(self) -> "_RVVisView":
        """Visualization view: (steps, 3, bodies)"""
        return self._r_vis_view

    @property
    def v_vis(self) -> "_RVVisView":
        """Visualization view: (steps, 3, bodies)"""
        return self._v_vis_view

    @property
    def t(self) -> FloatArray:
        if self._t is None:
            return np.arange(self.steps) * self.dt
        return self._t


class _RVView:
    def __init__(self, parent: "SimstateMemmap", rv: Literal["r", "v"]) -> None:
        self._parent = parent
        self._rv = rv

    def __getitem__(self, key: Index | Tuple[Index, ...]) -> FloatArray:
        data = self._parent.mm
        last_dim = slice(0, 3) if self._rv == "r" else slice(3, 6)

        step: Index
        body: Index

        # Normalize key into (step, body)
        if isinstance(key, tuple):
            # Now mypy knows key is a tuple and slicing is allowed
            if len(key) == 1:
                step, body = key[0], slice(None)
            else:
                step, body = key[0], key[1]
        else:
            step, body = key, slice(None)

        # Promote ints to slices for safe indexing
        if isinstance(step, int):
            step = slice(step, step + 1)
        if isinstance(body, int):
            body = slice(body, body + 1)

        return data[step, body, last_dim]


class _RVVisView:
    def __init__(self, base_view: _RVView) -> None:
        self._base = base_view

    def __getitem__(self, key: Index | Tuple[Index, ...]) -> FloatArray:
        # base returns (steps, bodies, 3)
        data = self._base[key]
        return np.transpose(data, (0, 2, 1))  # (steps, 3, bodies)
