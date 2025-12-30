"""Module for Horizons API"""

import json
import os
from hashlib import sha1
from io import StringIO
from typing import Dict

import numpy as np
import requests

from project.utils import Dir, FloatArray, T
from project.utils.horizons.const import Body, HorizonsParams, bodies
from project.utils.time import TimeConvert


class Horizons:
    """Horizons API class"""

    URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

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
        params_hash = sha1(repr(json.dumps(params, sort_keys=True)).encode())
        cache_path = Dir.horizons / f"horizons_{params_hash.hexdigest()}.txt"

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
    def retrieve_pos(body_name: Body, time_jd: float) -> FloatArray:
        """Retrieve body position in Geocentric inertial frame

        Parameters
        ----------
        body_name : Body
            Name of body (see Horizons API name scheme)
        time_jd : float
            JD for wanted time

        Returns
        -------
        A
            Body position vector (3,)
        """
        data = Horizons.get(
            params={
                "format": "text",
                "COMMAND": f"'{Horizons.body(body_name)}'",
                "EPHEM_TYPE": "'VECTORS'",
                "CENTER": "'500@399'",
                "REF_PLANE": "'FRAME'",
                "START_TIME": f"'{TimeConvert.cal2str(TimeConvert.jd2cal(time_jd))}'",
                "STOP_TIME": f"'{TimeConvert.cal2str(TimeConvert.jd2cal(time_jd + T.s / T.d))}'",
                "VEC_TABLE": "'1'",
                "CSV_FORMAT": "'YES'",
                "VEC_LABELS": "'NO'",
            }
        )
        csv_data = data.split("$$SOE")[1].split("$$EOE")[0]

        return np.genfromtxt(StringIO(csv_data), delimiter=",", usecols=(2, 3, 4))

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
        if body_name not in bodies:
            raise ValueError("Body not present in Horizons System")
        return bodies[body_name]

    @staticmethod
    def body_dict() -> Dict[str, str]:
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


if __name__ == "__main__":
    sun_p = Horizons.retrieve_pos("Sun", 2460903.5)
    print(sun_p)
