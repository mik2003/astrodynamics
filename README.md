<!--
SPDX-FileCopyrightText: 2026 Michelangelo Secondo <michelangelo@secondo.aero>

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Astrodynamics

A small but structured Python package for **orbital dynamics simulation**, **ephemeris handling**, and **trajectory visualization**, intended for scientific, educational, and prototyping use.

The project supports simple N-body propagation, Horizons-based ephemeris ingestion, optional C++ acceleration, and matplotlib-based visualization. It is fully type-annotated and uses a modern `src/` layout.

---

## Highlights

- **Package name:** `project`
- **Layout:** `src/`-based, PEP 621 compliant
- **Typing:** fully typed (`py.typed` included)
- **Domains:** astrodynamics, N-body simulation, ephemerides
- **Acceleration:** optional C++ force kernel via `pybind11`
- **Audience:** education, experimentation, early-stage research

---

## Repository layout

### Top level

```text
.
├─ pyproject.toml          # Project metadata, dependencies, tooling config
├─ README.md               # This file
├─ LICENSE
├─ build.bat               # Helper script for native build (Windows)
├─ data/                   # Sample ephemeris and system definition files
├─ profiling/              # Profiling utilities and results
├─ src/                    # Source tree (Python + C++)
├─ test/                   # Unit tests
└─ cache/                  # Local scratch / experiment outputs
```

### Python package (`src/project/`)

```txt
src/project/
├─ __init__.py
├─ __main__.py             # Example entry point (python -m project)
├─ _version.py
├─ py.typed                # Marks package as typed
│
├─ simulation/             # Core simulation logic
│  ├─ integrals.py         # Analytical integrals / helpers
│  ├─ integrator.py        # Time integration schemes
│  ├─ model.py             # Physical and numerical models
│  ├─ propagator.py        # High-level propagation orchestration
│  └─ cpp_force_kernel/    # Python bindings to optional C++ backend
│
├─ utils/                  # Supporting utilities
│  ├─ data.py              # Data containers and helpers
│  ├─ simstate.py          # Simulation state representations
│  ├─ siminteg.py          # Integration helpers
│  ├─ time.py              # Time handling utilities
│  └─ horizons/            # NASA Horizons interface
│     ├─ const.py
│     └─ ephemeris.py
│
├─ ui/                     # Visualization & UI helpers
│  ├─ constants.py
│  └─ elements.py
│
└─ temp/                   # Experiments, demos, and generated media
   ├─ 3bp.py
   ├─ exercise.py
   ├─ fictional_ephemeris.py
   ├─ horizon.py
   └─ *.gif
```

### Native extension (`src/cpp/`)

```txt
src/cpp/
├─ CMakeLists.txt
├─ main.cpp                # Force kernel implementation
└─ include/                # vendored headers (pybind11, xtensor, etc.)
```
The C++ component provides an optional performance boost for force evaluation and is **not required** for basic usage.

### Data files (`data/`)

Sample TOML system definitions used for simulations:

`sun_earth_moon_2460966.toml`
`inner_solar_system_2460959.toml`
`solar_system_2460967.toml`
`solar_system_moons_2460966.toml`
`figure-8.toml`
`fictional_system_2460959.toml`

A small conversion helper (`json2toml.py`) is included for legacy formats.

### Profiling (`profiling/`)

`profile_propagator.py` — profiling harness

`profile_stats.py` — summary and analysis

`profiling/README.md` — usage notes

The package is instrumented to support performance analysis during development.

### Tests (`test/`)

`test_simulation.py`

`test_pointmass_derivative_consistency.py`

Tests focus on physical consistency and numerical correctness.

## Installation

The project targets Python 3.13.

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -e .[dev,test]
```


Editable installs are recommended during development.

## Quick start

Run the example entry point:

```powershell
py -m project
```


This sets up a simple simulation using one of the bundled datasets and runs a propagation / visualization pipeline.

## Example usage
### Load a system definition

```python
from pathlib import Path
from project.utils.data import BodyList

bl = BodyList.load(Path("data/sun_earth_moon_2460966.toml"))
print([b.name for b in bl])
print(bl.r_0.shape)
```

### Fetch ephemerides from NASA Horizons

```python
from project.utils.horizons.ephemeris import horizons_request, safe_json_parse

resp = horizons_request(
    command="399",
    start_time="2025-01-01",
    stop_time="2025-01-02",
    step_size="1d",
)

parsed = safe_json_parse(resp.text)
```


⚠️ Use responsibly: respect Horizons rate limits and include identifying information when appropriate.

## Design notes

- **Numerical focus**: clarity and correctness over extreme performance

- **Strong typing**: all public modules are type-annotated

- **Modularity**: simulation, utilities, and UI are cleanly separated

- **Extensibility**: C++ backend can be swapped or extended independently

## License

See LICENSE for details.
