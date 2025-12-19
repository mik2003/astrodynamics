"""Time utilities module"""

from dataclasses import dataclass
from typing import Tuple, overload

import numpy as np

from project.utils import A, A_int, floatA, intA


@dataclass(frozen=True)
class T:
    """Time in [s]"""

    s = 1.0  # Second
    m = 60.0  # Minute
    h = 3600.0  # Hour
    d = 86400.0  # Day
    a = 31557600.0  # Annum


@dataclass(frozen=True)
class TimeConvert:
    """Class for time operations"""

    mjd_shift = 2400000.5  # [d]
    DATE_STR = "{}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}"

    @overload
    @staticmethod
    def d2jd(d: float, epoch: float) -> float: ...
    @overload
    @staticmethod
    def d2jd(d: A, epoch: float) -> A: ...
    @staticmethod
    def d2jd(d: floatA, epoch: float) -> floatA:
        """Days to julian date (JD)

        Parameters
        ----------
        d : float or numpy array of floats
            Days since epoch
        epoch : float
            Epoch in julian date (JD)

        Returns
        -------
        same type as d
            Julian date (JD)
        """
        return d + epoch

    @overload
    @staticmethod
    def s2jd(s: float, epoch: float) -> float: ...
    @overload
    @staticmethod
    def s2jd(s: A, epoch: float) -> A: ...
    @staticmethod
    def s2jd(s: floatA, epoch: float) -> floatA:
        """Seconds to julian date (JD)

        Parameters
        ----------
        s : float or numpy array of floats
            Seconds since epoch
        epoch : float
            Epoch in julian date (JD)

        Returns
        -------
        same type as d
            Julian date (JD)
        """
        return TimeConvert.d2jd(d=s / T.d, epoch=epoch)

    @overload
    @staticmethod
    def jd2mjd(jd: float) -> float: ...
    @overload
    @staticmethod
    def jd2mjd(jd: A) -> A: ...
    @staticmethod
    def jd2mjd(jd: floatA) -> floatA:
        """Julian date (JD) to modified julian date (MJD)

        Parameters
        ----------
        jd : float or numpy array of floats
            Julian date (JD)

        Returns
        -------
        same type as jd
            Modified julian date (MJD)
        """
        return jd - TimeConvert.mjd_shift

    @overload
    @staticmethod
    def d2mjd(d: float, epoch: float) -> float: ...
    @overload
    @staticmethod
    def d2mjd(d: A, epoch: float) -> A: ...
    @staticmethod
    def d2mjd(d: floatA, epoch: float) -> float | A:
        """Days to modified julian date (MJD)

        Parameters
        ----------
        d : float or numpy array of floats
            Days since epoch
        epoch : float
            Epoch in julian date (JD)

        Returns
        -------
        same type as d
            Modified julian date (MJD)
        """
        return TimeConvert.jd2mjd(jd=TimeConvert.d2jd(d=d, epoch=epoch))

    @overload
    @staticmethod
    def s2mjd(s: float, epoch: float) -> float: ...
    @overload
    @staticmethod
    def s2mjd(s: A, epoch: float) -> A: ...
    @staticmethod
    def s2mjd(s: floatA, epoch: float) -> floatA:
        """Days to modified julian date (MJD)

        Parameters
        ----------
        s : float or numpy array of floats
            Seconds since epoch
        epoch : float
            Epoch in julian date (JD)

        Returns
        -------
        same type as s
            Modified julian date (MJD)
        """
        return TimeConvert.d2mjd(d=s / T.d, epoch=epoch)

    @overload
    @staticmethod
    def jd2cal(jd: float) -> Tuple[int, int, int, int, int, int]: ...
    @overload
    @staticmethod
    def jd2cal(
        jd: A,
    ) -> Tuple[A_int, A_int, A_int, A_int, A_int, A_int]: ...
    @staticmethod
    def jd2cal(jd: floatA) -> Tuple[intA, intA, intA, intA, intA, intA]:
        """Method to convert julian date (JD) to calendar date/time
        Only works for dates after the implementation of the
        gregorian calendar in 1582 (1583 on is safe)

        Parameters
        ----------
        jd : float
            Julian date (JD)

        Returns
        -------
        Tuple[int]
            YYYY, MM, DD, hh, mm, ss
        """
        jd0 = jd + 0.5
        l1 = np.trunc(jd0 + 68569)
        l2 = np.trunc(4 * l1 / 146097)
        l3 = l1 - np.trunc((146097 * l2 + 3) / 4)
        l4 = np.trunc(4000 * (l3 + 1) / 1461001)
        l5 = l3 - np.trunc(1461 * l4 / 4) + 31
        l6 = np.trunc(80 * l5 / 2447)
        l7 = np.trunc(l6 / 11)

        year = 100 * (l2 - 49) + l4 + l7
        month = l6 + 2 - 12 * l7
        day = l5 - np.trunc(2447 * l6 / 80)

        # Rounding is consistent with https://ssd.jpl.nasa.gov/tools/jdc
        s = np.round(np.remainder(jd0, 1) * T.d)

        hours = np.remainder(s / T.h, 24)
        minutes = np.remainder(s / T.m, 60)
        seconds = np.remainder(s / T.s, 60)

        return (
            np.astype(year, int),
            np.astype(month, int),
            np.astype(day, int),
            np.astype(hours, int),
            np.astype(minutes, int),
            np.astype(seconds, int),
        )

    @overload
    @staticmethod
    def cal2jd(cal: Tuple[int, int, int, int, int, int]) -> float: ...
    @overload
    @staticmethod
    def cal2jd(cal: Tuple[A_int, A_int, A_int, A_int, A_int, A_int]) -> A: ...
    @staticmethod
    def cal2jd(cal: Tuple[intA, intA, intA, intA, intA, intA]) -> floatA:
        """Method to convert calendar date/time to julian date (JD)
        Only works for dates after the implementation of the
        gregorian calendar in 1582 (1583 on is safe)

        Parameters
        ----------
        cal : Tuple[int]
            YYYY, MM, DD, hh, mm, ss

        Returns
        -------
        float
            Julian date (JD)
        """
        y, m, d, hours, minutes, seconds = cal
        f = (seconds + minutes * T.m + hours * T.h) / T.d
        c = np.trunc((m - 14) / 12)
        jd0 = (
            d
            - 32075
            + np.trunc(1461 * (y + 4800 + c) / 4)
            + np.trunc(367 * (m - 2 - c * 12) / 12)
            - np.trunc(3 * (np.trunc((y + 4900 + c) / 100)) / 4)
        )
        jd = jd0 + f - 0.5

        return jd  # type: ignore

    @overload
    @staticmethod
    def s2cal(s: float, epoch: float) -> Tuple[int, int, int, int, int, int]: ...
    @overload
    @staticmethod
    def s2cal(
        s: A, epoch: float
    ) -> Tuple[A_int, A_int, A_int, A_int, A_int, A_int]: ...
    @staticmethod
    def s2cal(s: floatA, epoch: float) -> Tuple[intA, intA, intA, intA, intA, intA]:
        """Seconds to calendar date/time

        Parameters
        ----------
        s : float or numpy array of floats
            Seconds since epoch
        epoch : float
            Epoch in julian date (JD)

        Returns
        -------
        Tuple[int]
            YYYY, MM, DD, hh, mm, ss
        """
        return TimeConvert.jd2cal(jd=TimeConvert.s2jd(s=s, epoch=epoch))

    @staticmethod
    def cal2str(cal: Tuple[int, int, int, int, int, int]) -> str:
        return TimeConvert.DATE_STR.format(*cal)
