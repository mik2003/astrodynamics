from project.utils.body_registry import BodyID


class BodyPresets:
    sun_earth_moon: list[BodyID] = [
        BodyID.SUN,
        BodyID.EARTH,
        BodyID.MOON,
    ]

    inner_solar_system: list[BodyID] = [
        BodyID.SUN,
        BodyID.MERCURY,
        BodyID.VENUS,
        BodyID.EARTH,
        BodyID.MOON,
        BodyID.MARS,
    ]

    inner_solar_system_jupiter: list[BodyID] = [
        BodyID.SUN,
        BodyID.MERCURY,
        BodyID.VENUS,
        BodyID.EARTH,
        BodyID.MOON,
        BodyID.MARS,
        BodyID.JUPITER,
    ]

    solar_system: list[BodyID] = [
        BodyID.SUN,
        BodyID.MERCURY,
        BodyID.VENUS,
        BodyID.EARTH,
        BodyID.MOON,
        BodyID.MARS,
        BodyID.JUPITER,
        BodyID.SATURN,
        BodyID.URANUS,
        BodyID.NEPTUNE,
    ]

    solar_system_dwarf: list[BodyID] = [
        BodyID.SUN,
        BodyID.MERCURY,
        BodyID.VENUS,
        BodyID.EARTH,
        BodyID.MOON,
        BodyID.MARS,
        BodyID.JUPITER,
        BodyID.SATURN,
        BodyID.URANUS,
        BodyID.NEPTUNE,
        # Dwarf planets
        BodyID.N_1_CERES,
        BodyID.PLUTO,
        BodyID.N_136108_HAUMEA,
        BodyID.N_136472_MAKEMAKE,
        BodyID.N_136199_ERIS,
    ]

    solar_system_moons: list[BodyID] = [
        BodyID.SUN,
        BodyID.MERCURY,
        BodyID.VENUS,
        BodyID.EARTH,
        BodyID.MOON,
        BodyID.MARS,
        BodyID.PHOBOS,
        BodyID.DEIMOS,
        BodyID.JUPITER,
        BodyID.IO,
        BodyID.EUROPA,
        BodyID.GANYMEDE,
        BodyID.CALLISTO,
        BodyID.AMALTHEA,
        BodyID.HIMALIA,
        BodyID.THEBE,
        BodyID.ADRASTEA,
        BodyID.METIS,
        BodyID.SATURN,
        BodyID.MIMAS,
        BodyID.ENCELADUS,
        BodyID.TETHYS,
        BodyID.DIONE,
        BodyID.RHEA,
        BodyID.TITAN,
        BodyID.HYPERION,
        BodyID.IAPETUS,
        BodyID.PHOEBE,
        BodyID.JANUS,
        BodyID.EPIMETHEUS,
        BodyID.HELENE,
        BodyID.ATLAS,
        BodyID.PROMETHEUS,
        BodyID.PANDORA,
        BodyID.PAN,
        BodyID.URANUS,
        BodyID.ARIEL,
        BodyID.UMBRIEL,
        BodyID.TITANIA,
        BodyID.OBERON,
        BodyID.MIRANDA,
        BodyID.NEPTUNE,
        BodyID.TRITON,
        BodyID.NEREID,
        BodyID.NAIAD,
        BodyID.THALASSA,
        BodyID.DESPINA,
        BodyID.GALATEA,
        BodyID.LARISSA,
        BodyID.PROTEUS,
        BodyID.PLUTO,
        BodyID.CHARON,
        BodyID.NIX,
        BodyID.HYDRA,
        BodyID.KERBEROS,
        BodyID.STYX,
        BodyID.N_1_CERES,
        BodyID.N_136108_HAUMEA,
        BodyID.N_136472_MAKEMAKE,
        BodyID.N_136199_ERIS,
    ]


if __name__ == "__main__":
    import time

    from project.utils.apis.horizons import HorizonsBodyYear

    t0 = time.time()

    for body in BodyPresets.solar_system_moons:
        print(f"Retrieving {body}...")
        b = HorizonsBodyYear(body, 2026)
        try:
            print("Mean radius", b.radius)
            print("GM", b.gm)
        except ValueError as e:
            print(e)

    print(f"Time elapsed: {time.time() - t0:.2f} [s]")
