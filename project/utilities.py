import os
import pathlib
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np
import numpy.typing as npt

A = npt.NDArray[np.float64]


@dataclass(frozen=True)
class T:
    """Time in [s]"""

    s = 1.0  # Second
    m = 60.0  # Minute
    h = 3600.0  # Hour
    d = 86400.0  # Day
    a = 31557600.0  # Annum


@dataclass(frozen=True)
class D:
    """Distances in [m]"""

    au = 149597870700  # Astronomical unit
    ly = 9460730472580800  # Light year
    pc = 96939420213600000 / np.pi  # Parsec


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


def datetime_to_jd(dt: datetime) -> float:
    """Convert datetime to Julian Date"""
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second

    # Convert to fractional day
    fractional_day = (hour + minute / 60 + second / 3600) / 24

    # Julian Date calculation
    if month <= 2:
        year -= 1
        month += 12

    a = year // 100
    b = 2 - a + a // 4
    jd_day = (
        int(365.25 * (year + 4716))
        + int(30.6001 * (month + 1))
        + day
        + b
        - 1524.5
    )

    return jd_day + fractional_day


def print_progress(i: int, n: int, start_time: float):
    progress = int(i / n * 50)
    bar = "[" + "#" * progress + "-" * (50 - progress) + "]"
    elapsed = time.time() - start_time
    if i > 0:
        est_total = elapsed / i * n
        est_remain = est_total - elapsed
        hrs = int(est_remain // 3600)
        mins = int((est_remain % 3600) // 60)
        secs = int(est_remain % 60)
        est_str = f" | ETA: {hrs:02d}:{mins:02d}:{secs:02d}"
    else:
        est_str = ""
    print(f"\rProgress {bar} {i/n*100:.2f}%{est_str}", end="")


def print_done() -> None:
    print(
        "\rProgress [" + "#" * 50 + "] 100.00% | ETA: 00:00:00", end=""
    )  # Show full bar at the end


class ValueUnitToStr:
    @staticmethod
    def s(value: float, str_format: str = "{:.2f}") -> str:
        """Seconds"""
        if value >= T.a:
            return str_format.format(value / T.a) + " a"
        elif value >= T.d:
            return str_format.format(value / T.d) + " d"
        elif value >= T.h:
            return str_format.format(value / T.h) + " h"
        elif value >= T.m:
            return str_format.format(value / T.m) + " m"
        else:
            return str_format.format(value) + " s"

    @staticmethod
    def s_per_s(value: float, str_format: str = "{:.2f}") -> str:
        """Seconds"""
        if value >= T.a:
            return str_format.format(value / T.a) + " years/s"
        elif value >= T.d:
            return str_format.format(value / T.d) + " days/s"
        elif value >= T.h:
            return str_format.format(value / T.h) + " hours/s"
        elif value >= T.m:
            return str_format.format(value / T.m) + " mins/s"
        else:
            return str_format.format(value) + " s/s"

    @staticmethod
    def m(value: float, str_format: str = "{:.2e}") -> str:
        """Meters"""
        # if value >= D.pc:
        #     return str_format.format(value / D.pc) + " pc"
        if value >= D.ly:
            return str_format.format(value / D.ly) + " ly"
        elif value >= D.au:
            return str_format.format(value / D.au) + " AU"
        elif value >= 1e3:
            return str_format.format(value / 1e3) + " km"
        elif value >= 1:
            return str_format.format(value / 1) + " m"
        # elif value >= 1e-1:
        #     return str_format.format(value / 1e-1) + " dm"
        # elif value >= 1e-2:
        #     return str_format.format(value / 1e-2) + " cm"
        else:
            return str_format.format(value / 1e-3) + " mm"

    @staticmethod
    def m_per_px(value: float, str_format: str = "{:.2e}") -> str:
        """Meters_per_px"""
        return ValueUnitToStr.m(value=value, str_format=str_format) + "/px"
