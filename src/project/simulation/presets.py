from typing import List

from project.utils.horizons import Body


class BodySystemPreset:
    sun_earth_moon: List[Body] = [
        "Sun",
        "Earth",
        "Moon",
    ]

    inner_solar_system: List[Body] = [
        "Sun",
        "Mercury",
        "Venus",
        "Earth",
        "Moon",
        "Mars",
    ]

    inner_solar_system_jupiter: List[Body] = [
        "Sun",
        "Mercury",
        "Venus",
        "Earth",
        "Moon",
        "Mars",
        "Jupiter",
    ]

    # Main planets (traditional solar system)
    solar_system: List[Body] = [
        "Sun",
        "Mercury",
        "Venus",
        "Earth",
        "Moon",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
    ]

    # Main planets + dwarf planets
    solar_system_dwarf: List[Body] = [
        "Sun",
        "Mercury",
        "Venus",
        "Earth",
        "Moon",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
        # Dwarf planets
        "Ceres",
        "Pluto",
        "Haumea",
        "Makemake",
        "Eris",
    ]

    # Main planets + major moons
    solar_system_moons: List[Body] = [
        "Sun",
        "Mercury",
        "Venus",
        # Earth system
        "Earth",
        "Moon",
        # Mars system
        "Mars",
        "Phobos",
        "Deimos",
        # Jupiter system
        "Jupiter",
        "Io",
        "Europa",
        "Ganymede",
        "Callisto",
        "Amalthea",
        "Himalia",
        "Thebe",
        "Adrastea",
        "Metis",
        # Saturn system
        "Saturn",
        "Mimas",
        "Enceladus",
        "Tethys",
        "Dione",
        "Rhea",
        "Titan",
        "Hyperion",
        "Iapetus",
        "Phoebe",
        "Janus",
        "Epimetheus",
        "Helene",
        "Atlas",
        "Prometheus",
        "Pandora",
        "Pan",
        # Uranus system
        "Uranus",
        "Ariel",
        "Umbriel",
        "Titania",
        "Oberon",
        "Miranda",
        # Neptune system
        "Neptune",
        "Triton",
        "Nereid",
        "Naiad",
        "Thalassa",
        "Despina",
        "Galatea",
        "Larissa",
        "Proteus",
        # Pluto system
        "Pluto",
        "Charon",
        "Nix",
        "Hydra",
        "Kerberos",
        "Styx",
        # Dwarf planets
        "Charon",
        "Ceres",
        "Eris",
        "Haumea",
        "Makemake",
    ]


if __name__ == "__main__":
    # import time
    # from typing import List

    # from project.utils.horizons import HorizonsBodyYear

    # t0 = time.time()

    # for body in BodySystemPreset.solar_system_moons:
    #     print(f"Retrieving {body}...")
    #     b = HorizonsBodyYear(body, 2025)
    #     try:
    #         print("Mean radius", b.radius)
    #         print("GM", b.gm)
    #     except ValueError as e:
    #         print(e)

    # print(f"Time elapsed: {time.time() - t0:.2f} [s]")

    print(hasattr(BodySystemPreset, "solar_system_moons"))
