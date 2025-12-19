import json

import tomli_w


def json2toml(filename: str) -> None:
    with open(filename + ".json", "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(filename + ".toml", "wb") as f:
        tomli_w.dump(data, f)


if __name__ == "__main__":
    json2toml("sun_earth_moon_2460966")
