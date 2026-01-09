"""Module for Horizons API"""

import json
import os
import re
import tomllib
from datetime import datetime
from hashlib import sha1
from io import StringIO
from typing import Any, Dict, List, Literal, NotRequired, Tuple, TypedDict

import numpy as np
import requests
import tomli_w

from project.utils import Dir, FloatArray, T
from project.utils.apis.systeme_solaire import SystemeSolaireBodies
from project.utils.body_registry import BODY_REGISTRY, BodyID
from project.utils.time_utils import TimeConvert


class HorizonsParams(TypedDict):
    """Parameters for the Horizons API"""

    # Common Parameters
    format: NotRequired[Literal["json", "text"]]  # json
    COMMAND: NotRequired[str]  # none
    OBJ_DATA: NotRequired[Literal["'NO'", "'YES'"]]  # YES
    MAKE_EPHEM: NotRequired[Literal["'NO'", "'YES'"]]  # YES
    EPHEM_TYPE: NotRequired[
        Literal["'OBSERVER'", "'VECTORS'", "'ELEMENTS'", "'SPK'", "'APPROACH'"]
    ]  # OBSERVER
    EMAIL_ADDR: NotRequired[str]  # none
    # Ephemeris-Specific Parameters
    CENTER: NotRequired[str]  # Geocentric
    REF_PLANE: NotRequired[
        Literal["'ECLIPTIC'", "'FRAME'", "'BODY EQUATOR'"]
    ]  # ECLIPTIC
    COORD_TYPE: NotRequired[Literal["'GEODETIC'", "'CYLINDRICAL'"]]  # GEODETIC
    SITE_COORD: NotRequired[str]  # '0,0,0'
    START_TIME: NotRequired[str]  # none
    STOP_TIME: NotRequired[str]  # none
    STEP_SIZE: NotRequired[str]  # '60 min'
    TIME_DIGITS: NotRequired[Literal["'MINUTES'", "'SECONDS'", "'FRACSEC'"]]  # MINUTES
    TIME_TYPE: NotRequired[Literal["'UT'", "'TT'", "'TDB'"]]  # varies with EPHEM_TYPE
    TIME_ZONE: NotRequired[str]  # '+00:00'
    TLIST: NotRequired[str]  # none
    TLIST_TYPE: NotRequired[Literal["'JD'", "'MJD'", "'CAL'"]]  # none
    QUANTITIES: NotRequired[Literal["'A'"]]  # 'A'
    REF_SYSTEM: NotRequired[Literal["'ICRF'", "'B1950'"]]  # ICRF
    OUT_UNITS: NotRequired[Literal["'KM-S'", "'AU-D'", "'KM-D'"]]  # KM-S
    VEC_TABLE: NotRequired[
        Literal[
            "'1'",
            "'1x'",
            "'1xa'",
            "'1xar'",
            "'1xarp'",
            "'2'",
            "'2x'",
            "'2xa'",
            "'2xar'",
            "'2xarp'",
            "'3'",
            "'4'",
            "'5'",
            "'6'",
        ]
    ]  # 3
    VEC_CORR: NotRequired[Literal["'NONE'", "'LT'", "'LT+S'"]]  # NONE
    CAL_FORMAT: NotRequired[Literal["'CAL'", "'JD'", "'BOTH'"]]  # CAL
    CAL_TYPE: NotRequired[Literal["'MIXED'", "'GREGORIAN'"]]  # MIXED
    ANG_FORMAT: NotRequired[Literal["'HMS'", "'DEG'"]]  # HMS
    APPARENT: NotRequired[Literal["'AIRLESS'", "'REFRACTED'"]]  # AIRLESS
    RANGE_UNITS: NotRequired[Literal["'AU'", "'KM'"]]  # AU
    # ... TODO
    CSV_FORMAT: NotRequired[Literal["'NO'", "'YES'"]]  # NO
    VEC_LABELS: NotRequired[Literal["'NO'", "'YES'"]]  # YES
    VEC_DELTA_T: NotRequired[Literal["'NO'", "'YES'"]]  # NO
    # ... TODO


class Horizons:
    """Horizons API class"""

    URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
    BodyNamesCache = Dir.horizons / "horizons__body_names.toml"

    @staticmethod
    def get(params: HorizonsParams) -> str:
        """GET request to Horizons API

        Parameters
        ----------
        params : HorizonsParams
            Parameters of the request

        Returns
        -------
        str
            Horizons data
        """
        # Create unique hash from parameters to identify cache
        params_hash = sha1(
            repr(json.dumps(params, sort_keys=True)).encode()
        ).hexdigest()
        cache_path = Dir.horizons / f"horizons_{params_hash}.txt"

        if os.path.exists(cache_path):
            # Horizons data already in cache
            with open(cache_path, "r", encoding="utf-8") as f:
                out = f.read()
        else:
            # Request data through Horizons API
            r = requests.get(
                Horizons.URL,
                params=params,  # type: ignore
                timeout=5,
            )
            out = r.text

            # Save data to cache
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(out)

        return out

    @staticmethod
    def retrieve_body_year(body: BodyID, year: int) -> str:
        """Retrieve body position in SSB (Solar System Barycenter) inertial frame
        for an entire year, with 1 h sampling

        Parameters
        ----------
        body : BodyID
            ID of body
        year : int
            Year

        Returns
        -------
        str
            Horizons data
        """
        horizon_id = BODY_REGISTRY[body].horizons_id
        if len(horizon_id) == 8:
            command = f"'DES={horizon_id};'"
        else:
            command = f"'{horizon_id}'"

        return Horizons.get(
            params={
                "format": "text",
                "COMMAND": command,
                "EPHEM_TYPE": "'VECTORS'",
                "CENTER": "'500@0'",
                "REF_PLANE": "'FRAME'",
                "START_TIME": f"'{TimeConvert.cal2str((year, 1, 1, 0, 0, 0))}'",
                "STOP_TIME": f"'{TimeConvert.cal2str((year, 12, 31, 23, 59, 59))}'",
                "STEP_SIZE": "'1h'",
                "VEC_TABLE": "'2'",
                "CSV_FORMAT": "'YES'",
                "VEC_LABELS": "'NO'",
            }
        )

    @staticmethod
    def body(body_name: str) -> str:
        """Retrieve Horizons body ID

        Parameters
        ----------
        body_name : str
            Name of body (see Horizons for available bodies)

        Returns
        -------
        str
            Horizons body ID

        Raises
        ------
        ValueError
            If body not available in Horizons
        """
        # Retrieve list of bodies from Horizons and cache it
        if not Horizons.BodyNamesCache.exists():
            with open(Horizons.BodyNamesCache, "wb") as f:
                tomli_w.dump(Horizons.body_names(), f)

        # Retrieve body ID
        with open(Horizons.BodyNamesCache, "rb") as f:
            bodies: Dict[str, str] = tomllib.load(f)
        if body_name not in bodies:
            raise ValueError("Body not present in Horizons System")
        return bodies[body_name]

    @staticmethod
    def body_names() -> Dict[str, str]:
        """Retrieve dictionary of Horizons bodies and their IDs

        Returns
        -------
        Dict[str, str]
            Dictionary of bodies and respective IDs
        """
        out: Dict[str, str] = {}
        data = Horizons.get(params={"format": "text", "COMMAND": "'MB'"})
        data_lns = data.split("\n")[8:784]
        for ln in data_lns:
            out[ln[11:46].strip()] = ln[0:11].strip()

        return out


class HorizonsResponse:
    re_mean_radius = r"(?i:mean\sradius).*?=\s*([0-9]+(?:\.[0-9]+)?)"
    re_radius = r"(?i:radius).*?=\s*([0-9]+(?:\.[0-9]+)?)"
    re_rad = r"(?i:rad).*?=\s*([0-9]+(?:\.[0-9]+)?)"
    re_gm = r"GM.*?=\s*([0-9]+(?:\.[0-9]+)?)"
    re_mass = r"(?i:mass).*?=\s*([0-9]+(?:\.[0-9]+)?)"

    def __init__(self, text: str) -> None:
        self.text = text


class HorizonsBodyYear(HorizonsResponse):
    def __init__(self, body: BodyID, year: int) -> None:
        """Wrapper for Horizons response of single body single year

        Parameters
        ----------
        body : BodyID
            ID of body
        year : int
            Year of position
        """
        self.text = Horizons.retrieve_body_year(body=body, year=year)

        self.name = body
        self.year = year

        data_str = self.text.split("$$SOE")[1].split("$$EOE")[0]
        self.data = np.genfromtxt(
            StringIO(data_str), delimiter=",", usecols=(2, 3, 4, 5, 6, 7)
        )

        self._radius: float | None = None
        self._gm: float | None = None

    def get_state(self, month: int = 1, day: int = 1) -> FloatArray:
        """Retrieve body state in ICRF

        Parameters
        ----------
        body_name : Body
            Name of body (see Horizons API name scheme)
        year : int
            Year of position
        month : int, optional
            Month of position, by default January
        day : int, optional
            Day of position, by default the first of the month

        Returns
        -------
        A
            Body state vector (6,)
        """
        hours = (
            datetime(self.year, month, day) - datetime(self.year, 1, 1)
        ).total_seconds() / T.h

        return self.data[int(hours), :] * 1e3

    def get_pos(self, month: int = 1, day: int = 1) -> FloatArray:
        """Retrieve body position in ICRF

        Parameters
        ----------
        body_name : Body
            Name of body (see Horizons API name scheme)
        year : int
            Year of position
        month : int, optional
            Month of position, by default January
        day : int, optional
            Day of position, by default the first of the month

        Returns
        -------
        A
            Body position vector (3,)
        """
        return self.get_state(month=month, day=day)[:3]

    def get_vel(self, month: int = 1, day: int = 1) -> FloatArray:
        """Retrieve body velocity in ICRF

        Parameters
        ----------
        body_name : Body
            Name of body (see Horizons API name scheme)
        year : int
            Year of velocity
        month : int, optional
            Month of velocity, by default January
        day : int, optional
            Day of velocity, by default the first of the month

        Returns
        -------
        A
            Body velocity vector (3,)
        """
        return self.get_state(month=month, day=day)[3:]

    @property
    def radius(self) -> float:
        if self._radius is None:
            radius_lst = re.findall(HorizonsResponse.re_mean_radius, self.text)
            if len(radius_lst) == 0:
                radius_lst = re.findall(HorizonsResponse.re_radius, self.text)
            if len(radius_lst) == 0:
                radius_lst = re.findall(HorizonsResponse.re_rad, self.text)
            if len(radius_lst) == 0:
                raise ValueError(f"Radius not found for {self.name}")
            self._radius = float(radius_lst[0])
        return self._radius * 1e3

    @property
    def gm(self) -> float:
        if self._gm is None:
            gm_lst = re.findall(HorizonsResponse.re_gm, self.text)
            if len(gm_lst) == 0:
                raise ValueError(f"GM not found for {self.name}")
            self._gm = float(gm_lst[0])
        return self._gm * 1e9


def generate_sim_file(
    name: str, bodies: List[BodyID], time: Tuple[int, int, int]
) -> None:
    out: Dict[str, Any] = {
        "metadata": {
            "epoch": f"{time[0]:04d}-{time[1]:02d}-{time[2]:02d} 00:00:00",
            "target_count": len(bodies),
        },
        "body_list": [],
    }
    ssb = SystemeSolaireBodies()

    for body in bodies:
        hr = HorizonsBodyYear(body=body, year=time[0])
        y0 = hr.get_state(time[1], time[2])
        info = ssb.get_body_info(body)

        out["body_list"].append(
            {
                "name": body,
                "mu": info["gm"],
                "radius": info["radius"],
                "r_0": y0[:3].tolist(),
                "v_0": y0[3:].tolist(),
            }
        )

    with open(Dir.data / f"{name}.toml", "wb") as f:
        tomli_w.dump(out, f)

    print(f"{name}.toml")


if __name__ == "__main__":
    pass
    # print(Horizons.body("SSB"))
    # sun_p = Horizons.retrieve_pos("Sun", 2460903.5)
    # print(sun_p)
