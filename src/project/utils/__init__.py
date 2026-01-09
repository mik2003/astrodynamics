import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import EllipsisType
from typing import ParamSpec, Sequence, Union

import numpy as np

# Types
Float = float | np.floating
Int = int | np.integer
FloatArray = np.typing.NDArray[np.floating]  # Array type (floating)
IntArray = np.typing.NDArray[np.integer]  # Array type (integer)
# Union types for overloaded functions
FloatScalarOrArray = Float | FloatArray
IntScalarOrArray = Int | IntArray
# Indexing type
Index = Union[int, slice, Sequence[int], IntArray, EllipsisType, None]


P = ParamSpec("P")


@dataclass(frozen=True)
class C:
    """Universal constants"""

    G = 6.67430e-11  # m3/kg/s2
    G_KM = G * 1e-9


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
    """Paths relative to the project root."""

    # File where this class is defined
    _current_file = Path(__file__).resolve()

    # src/project/... -> project root = three levels up from src/project/utils
    root = _current_file.parents[3]

    # Directories
    data = root / "data"
    cache = root / "cache"
    simulation = cache / "simulation"
    horizons = cache / "horizons"
    test = cache / "test"
    secret = root / ".secret"

    # Files
    api_keys = secret / "api-keys.toml"


class ProgressTracker:
    def __init__(
        self,
        n: int,
        start_time: float | None = None,
        print_step: int = 10000,
        name: str = "Progress",
    ) -> None:
        """Print progress of a process with multiple steps

        Parameters
        ----------
        n : int
            Total number of steps
        start_time : float
            Initial time of process, initialization time by default
        print_step : int
            Number of steps between successive progress reports
        name : str
            Name of progress bar
        """
        self.n = n
        self.start_time = start_time or time.time()
        self.print_step = print_step
        self.name = name

    def print(self, i: int) -> None:
        """Print progress of a process with multiple steps

        Parameters
        ----------
        i : int
            Current step
        """
        if i == self.n:
            self.print_done()
        elif i % self.print_step == 0:
            progress = int(i / self.n * 50)
            bar = "[" + "#" * progress + "-" * (50 - progress) + "]"
            elapsed = self.elapsed
            if i > 0:
                est_total = elapsed / i * self.n
                est_remain = est_total - elapsed
                hrs = int(est_remain // 3600)
                mins = int((est_remain % 3600) // 60)
                secs = int(est_remain % 60)
                est_str = f" | ETA: {hrs:02d}:{mins:02d}:{secs:02d}"
            else:
                est_str = ""
            print(f"\r{self.name} {bar} {i / self.n * 100:.2f}%{est_str}", end="")

    def print_done(self) -> None:
        """Print process concluded"""
        print(
            f"\r{self.name} [" + "#" * 50 + f"] 100.00% | Time: {self.elapsed:.3f} s"
        )  # Show full bar at the end

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time


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
    jd_day = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5

    return jd_day + fractional_day


def print_progress(i: int, n: int, start_time: float) -> None:
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
    print(f"\rProgress {bar} {i / n * 100:.2f}%{est_str}", end="")


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
