# Astrodynamics

A small Python package for simple orbital simulation, ephemeris retrieval, and visualization utilities.

This repository contains tools to load body ephemerides (from local JSON or NASA Horizons), represent bodies and initial conditions, run time-domain simulations, and visualize trajectories. It also includes profiling outputs and helpers used during development.

## Highlights

- Python package name: `project` (packaged via `setup.py`).
- Intended for scientific/educational use: simple N-body integrations, ephemeris fetching, and plotting.
- Uses common scientific Python stack: `numpy`, `scipy`, `matplotlib`, `numba` (optional speed-ups), and `pydantic` for data validation.

## Repository layout

Top-level files:

- `Pipfile`, `pyproject.toml`, `setup.py` — project metadata and dependency hints.
- `README.md` — this file.
- `data/` — sample ephemeris JSON files used by simulations:
  - `inner_solar_system_2460959.json`
  - `solar_system_moons_2460966.json`
  - `sun_earth_moon_2460966.json`
  - `test.json`
- `profiling/` — profiling outputs and a small `profile_stats.py` to summarize runs.
- `project/` — main package code:
  - `__main__.py` — example runner that sets up a `Simulation` and `Visualization`.
  - `data.py` — data models for bodies and body lists (uses `pydantic` and `numpy`).
  - `ephemeris.py` — helpers to fetch and parse data (contains a NASA Horizons wrapper and constants).
  - `simulation.py` — simulation logic (N-body integrator).
  - `integrals.py`, `exercise.py`, `utilities.py`, `visualization.py` — supporting modules.

## Installation

The project targets Python 3.13 per `Pipfile`.

```powershell
# Using pipenv
pip install pipenv
pipenv install --dev
```

## Quick usage

Run the example simulation in `project.__main__`:

```powershell
python -m project
```

This will create a `Simulation` with one of the sample datasets and start the `Visualization`.

### Using modules directly

Load a body list from file and inspect initial states:

```python
from pathlib import Path
from project.data import BodyList

bl = BodyList.load(Path('data/sun_earth_moon_2460966.json'))
print([b.name for b in bl])
print('Initial positions shape:', bl.r_0.shape)
```

Fetch ephemeris from NASA Horizons (see `project.ephemeris.horizons_request`) — this function performs an HTTP request and returns the raw response. Use with care (respect rate limits and include an email address when appropriate):

```python
from project.ephemeris import horizons_request, safe_json_parse

resp = horizons_request(command='399', start_time='2025-01-01', stop_time='2025-01-02', step_size='1d')
parsed = safe_json_parse(resp.text)
```

## Key modules and responsibilities

- `project/data.py` — pydantic models for `Body` and `BodyList`. Validates vectors and provides convenience properties for stacked initial condition arrays.
- `project/ephemeris.py` — constants for body IDs and mu values, helper functions to request and parse Horizons output, and pre-defined target sets.
- `project/simulation.py` — contains `Simulation` class which orchestrates integration (time stepping, gravitational interactions). See the docstrings in the file for detailed parameters.
- `project/visualization.py` — plotting and visualization utilities; the example runner uses this to display trajectories.
- `profiling/` — contains profiling outputs and `profile_stats.py` to analyze runtime profiles.

## Data format

Sample JSON files in `data/` store a `body_list` key with an array of bodies. Each body includes:

- `name` — string
- `mu` — gravitational parameter (m^3/s^2)
- `r_0` — initial position vector [x, y, z] in meters
- `v_0` — initial velocity vector [vx, vy, vz] in meters/second

Use `BodyList.load(pathlib.Path('data/your_file.json'))` to load these files.

## Testing and profiling

Unit tests are not included in the repository root. The project contains profiling outputs in `profiling/` which record runtime behavior over different runs. Use `profiling/profile_stats.py` to summarize profiling files (see file header for usage).
