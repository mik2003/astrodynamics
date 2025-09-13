import pathlib


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case"""
    import re

    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


class Dir:
    """Directories."""

    main_dir = pathlib.Path(__file__).parent.absolute()
    in_dir = pathlib.Path(main_dir).parent.absolute().joinpath("in")
    out_dir = pathlib.Path(main_dir).parent.absolute().joinpath("out")
