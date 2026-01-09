import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from project.utils import Dir


def load_systeme_solaire(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {b["englishName"]: b for b in data["bodies"]}


def load_horizons(path: Path) -> Dict[str, str]:
    """
    Returns: { horizons_name : horizons_id }
    """
    with path.open("rb") as f:
        return tomllib.load(f)


def normalize_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


def extract_naif_id(value: str) -> str | None:
    # Horizons IDs are numeric strings
    if value.isdigit():
        return value
    return None


@dataclass(frozen=True)
class RegistryEntry:
    canonical_name: str
    horizons_id: str
    systeme_solaire_name: str


def generate_body_registry(
    horizons_toml: Path,
    systeme_solaire_json: Path,
) -> tuple[list[RegistryEntry], list[str]]:
    horizons = load_horizons(horizons_toml)
    ss = load_systeme_solaire(systeme_solaire_json)

    ss_by_norm = {normalize_name(name): name for name in ss}

    registry: list[RegistryEntry] = []
    unresolved: list[str] = []

    for h_name, h_id in horizons.items():
        naif_id = extract_naif_id(h_id)

        # Prefer exact NAIF match
        ss_match = None
        if naif_id:
            for ss_name, body in ss.items():
                if str(body.get("id")) == naif_id:
                    ss_match = ss_name
                    break

        # Fallback: normalized name match
        if ss_match is None:
            key = normalize_name(h_name)
            ss_match = ss_by_norm.get(key)

        # 🚨 Reject empty names explicitly
        if not ss_match:
            unresolved.append(h_name)
            continue

        if ss_match is None:
            unresolved.append(h_name)
            continue

        registry.append(
            RegistryEntry(
                canonical_name=ss_match,
                horizons_id=h_id,
                systeme_solaire_name=ss_match,
            )
        )

    return registry, unresolved


def emit_registry_py(
    registry: list[RegistryEntry],
    out_path: Path,
) -> None:
    lines = []
    lines.append("# AUTO-GENERATED — DO NOT EDIT\n")
    lines.append("from dataclasses import dataclass\n")
    lines.append("from enum import Enum\n\n\n")

    # Enum
    lines.append("class BodyID(str, Enum):\n")
    for r in registry:
        enum_name = r.canonical_name.upper().replace(" ", "_").replace("-", "_")
        lines.append(f'    {enum_name} = "{r.canonical_name}"\n')

    # Dataclass
    lines.append("\n@dataclass(frozen=True)\n")
    lines.append("class BodyInfo:\n")
    lines.append("    horizons_id: str\n")
    lines.append("    systeme_solaire_name: str\n\n")

    # Registry
    lines.append("BODY_REGISTRY = {\n")
    for r in registry:
        enum_name = r.canonical_name.upper().replace(" ", "_").replace("-", "_")
        lines.append(
            f"    BodyID.{enum_name}: "
            f'BodyInfo("{r.horizons_id}", "{r.systeme_solaire_name}"),\n'
        )
    lines.append("}\n")

    out_path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    registry, unresolved = generate_body_registry(
        horizons_toml=Dir.horizons / "horizons__body_names.toml",
        systeme_solaire_json=Dir.data / "systeme_solaire.json",
    )

    emit_registry_py(
        registry,
        out_path=Dir.data / "auto_body_registry.py",
    )

    if unresolved:
        print("⚠️ Unresolved bodies:")
        for name in unresolved:
            print("  -", name)
