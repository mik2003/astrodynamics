import json

import requests


def horizons_request(
    format: str = "json",
    command: str | None = None,
    obj_data: bool = True,
    make_ephem: bool = True,
    ephem_type: str = "VECTORS",
    email_addr: str | None = None,
) -> requests.Response:
    request = requests.get(
        "https://ssd.jpl.nasa.gov/api/horizons.api?"
        + f"format={format}"
        + f"{f"&COMMAND={command}" if command else ""}"
        + f"&OBJ_DATA={"YES" if obj_data else "NO"}"
        + f"&MAKE_EPHEM={"YES" if make_ephem else "NO"}"
        + f"&EPHEM_TYPE={ephem_type}"
        + f"{f"&EMAIL_ADDR={email_addr}" if email_addr else ""}",
        timeout=10,
    )
    return request


r = horizons_request(
    command="499", email_addr="michelangelosecondo+horizons@gmail.com"
)

jr = json.loads(r.content)
print(jr)
