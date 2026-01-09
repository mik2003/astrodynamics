import json
import tomllib
from typing import Any, Dict, List

import tomli_w

from project.utils import C, Dir

with Dir.api_keys.open("rb") as f:
    config = tomllib.load(f)
API_KEY = config["systeme-solaire"]
API_URL = "https://api.le-systeme-solaire.net/rest/bodies"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


class SystemeSolaireBodies:
    def __init__(self) -> None:
        with open(Dir.data / "systeme_solaire.json", "r", encoding="utf-8") as f:
            self.bodies = json.load(f)["bodies"]

    def extract_values(self) -> None:
        bodies_filtered: Dict[str, Dict[str, float | None]] = {}
        for body in self.bodies:
            bodies_filtered[body["englishName"]] = {
                "Radius": extract_radius(body),
                "GM": extract_gm(body),
            }
        with open(Dir.data / "systeme_solaire.toml", "wb") as f:
            tomli_w.dump(bodies_filtered, f)

    def extract_names(self) -> List[str]:
        out: List[str] = []
        for body in self.bodies:
            out.append(body["englishName"])

        return out

    def get_body_info(self, body_name: str) -> Dict[str, float]:
        """Retrieve radius (m) and GM (m^3/s^2) for a body from systeme-solaire."""
        for body in self.bodies:
            if body["englishName"] == body_name:
                return {
                    "radius": extract_radius(body),
                    "gm": extract_gm(body),
                }
        raise ValueError(f"Body {body_name} not found in systeme-solaire data")


def extract_radius(body: Dict[str, Any]) -> float:
    """Return mean radius in meters if available."""
    if body.get("meanRadius"):
        return float(body["meanRadius"]) * 1e3  # m
    return 0.0


def extract_gm(body: Dict[str, Any]) -> float:
    """Return GM in m^3/s^2 if possible."""
    # Some bodies provide mass instead of GM
    mass = body.get("mass", {})
    if (
        mass is not None
        and mass.get("massValue")
        and mass.get("massExponent") is not None
    ):
        mass_kg = float(mass["massValue"] * 10 ** mass["massExponent"])
        return C.G * mass_kg
    return 0.0


if __name__ == "__main__":
    pass
    # s = SystemeSolaireBodies()
    # n = s.extract_names()

    # with open("temp.txt", "w", encoding="utf-8") as f:
    #     f.writelines([f'"{body}"\n' for body in n])
