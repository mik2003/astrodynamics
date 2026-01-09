import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from project.utils import Dir

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_systeme_solaire(path: Path) -> Dict[str, Dict[str, Any]]:
    """Return bodies indexed by englishName."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {b["englishName"]: b for b in data["bodies"] if b.get("englishName")}


def load_horizons(path: Path) -> Dict[str, str]:
    """
    Returns:
        { horizons_name : horizons_command }
    where horizons_command is either an MB alias or NAIF ID
    """
    with path.open("rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


def infer_naif_from_name(name: str) -> str | None:
    """
    Infer Horizons command from Système Solaire englishName.

    Examples:
        "1 Ceres"        -> "20'000'001"
        "101955 Bennu"   -> "20'101'955"
    """
    first = name.split()[0]
    return str(int(first) + 20_000_000) if first.isdigit() else None


def enum_safe_name(name: str) -> str:
    """
    Convert a body name into a valid Python enum identifier.
    """
    cleaned = (
        name.upper()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )

    if cleaned[0].isdigit():
        return f"N_{cleaned}"

    return cleaned


# ---------------------------------------------------------------------------
# Registry model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryEntry:
    canonical_name: str  # Système Solaire englishName
    horizons_id: str  # Horizons COMMAND
    systeme_solaire_name: str  # Same as canonical_name (explicit)


# ---------------------------------------------------------------------------
# Registry generation
# ---------------------------------------------------------------------------


def generate_body_registry(
    horizons_toml: Path,
    systeme_solaire_json: Path,
) -> Tuple[List[RegistryEntry], List[str]]:
    horizons = load_horizons(horizons_toml)
    ss = load_systeme_solaire(systeme_solaire_json)

    ss_norm = {normalize_name(name): name for name in ss}

    registry: list[RegistryEntry] = []
    unresolved: list[str] = []

    # --- Pass 1: Horizons MB objects ---
    for h_name, h_cmd in horizons.items():
        ss_name = None

        # Exact englishName match
        if h_name in ss:
            ss_name = h_name

        # Normalized fallback
        if ss_name is None:
            ss_name = ss_norm.get(normalize_name(h_name))

        if ss_name is None:
            unresolved.append(h_name)
            continue

        registry.append(
            RegistryEntry(
                canonical_name=ss_name,
                horizons_id=h_cmd,
                systeme_solaire_name=ss_name,
            )
        )

    # --- Pass 2: Small bodies not in Horizons MB ---
    registered = {r.canonical_name for r in registry}

    for ss_name, body in ss.items():
        if ss_name in registered:
            continue

        if body.get("bodyType") not in {
            "Asteroid",
            "Dwarf Planet",
            "Comet",
        }:
            continue

        naif = infer_naif_from_name(ss_name)
        if naif is None:
            continue

        registry.append(
            RegistryEntry(
                canonical_name=ss_name,
                horizons_id=naif,
                systeme_solaire_name=ss_name,
            )
        )

    registry.sort(key=lambda r: r.canonical_name)
    return registry, unresolved


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def emit_registry_py(
    registry: List[RegistryEntry],
    out_path: Path,
) -> None:
    lines: list[str] = []

    lines.append("# AUTO-GENERATED — DO NOT EDIT\n")
    lines.append("from dataclasses import dataclass\n")
    lines.append("from enum import Enum\n\n\n")

    # Enum
    lines.append("class BodyID(str, Enum):\n")
    for r in registry:
        enum_name = enum_safe_name(r.canonical_name)
        lines.append(f'    {enum_name} = "{r.canonical_name}"\n')

    # Dataclass
    lines.append("\n@dataclass(frozen=True)\n")
    lines.append("class BodyInfo:\n")
    lines.append("    horizons_id: str\n")
    lines.append("    systeme_solaire_name: str\n\n")

    # Registry mapping
    lines.append("BODY_REGISTRY = {\n")
    for r in registry:
        enum_name = enum_safe_name(r.canonical_name)
        lines.append(
            f"    BodyID.{enum_name}: "
            f'BodyInfo("{r.horizons_id}", "{r.systeme_solaire_name}"),\n'
        )
    lines.append("}\n")

    out_path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    registry, unresolved = generate_body_registry(
        horizons_toml=Dir.horizons / "horizons__body_names.toml",
        systeme_solaire_json=Dir.data / "systeme_solaire.json",
    )

    emit_registry_py(
        registry,
        out_path=Dir.utils / "body_registry.py",
    )

    if unresolved:
        print("⚠️ Unresolved Horizons names:")
        for name in unresolved:
            print("  -", name)
