# project/fictional_solar_system.py

import json
import math
import pathlib
import random
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

from project.utils import Dir, datetime_to_jd


class FictionalSolarSystemGenerator:
    """
    Generates fictional solar system data with random bodies for testing and simulation.
    """

    def __init__(self, seed: int = None):
        """
        Initialize the generator with optional random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)

        # Realistic ranges for celestial bodies (in SI units)
        self.mu_range = (1e10, 1e20)  # Gravitational parameter (m³/s²)
        self.orbit_radius_range = (1e9, 1e13)  # Distance from center (m)
        self.velocity_range = (1e2, 1e5)  # Orbital velocity (m/s)
        self.body_mass_range = (1e18, 1e25)  # Mass (kg) - derived from mu
        self.mu_range = (1e10, 1e20)  # Gravitational parameter (m³/s²)
        self.orbit_radius_range = (1e10, 1e11)  # Distance from center (m)
        self.velocity_range = (1e2, 1e5)  # Orbital velocity (m/s)
        self.body_mass_range = (1e18, 1e24)  # Mass (kg) - derived from mu

        # Common celestial body names for inspiration
        self.name_prefixes = [
            "Alpha",
            "Beta",
            "Gamma",
            "Delta",
            "Epsilon",
            "Zeta",
            "Eta",
            "Theta",
            "Iota",
            "Kappa",
            "Lambda",
            "Mu",
            "Nu",
            "Xi",
            "Omicron",
            "Pi",
            "Rho",
            "Sigma",
            "Tau",
            "Upsilon",
            "Phi",
            "Chi",
            "Psi",
            "Omega",
        ]

        self.name_suffixes = [
            "Prime",
            "Secunda",
            "Tertia",
            "Quarta",
            "Quinta",
            "Sexta",
            "Septima",
            "Octava",
            "Nona",
            "Decima",
            "Major",
            "Minor",
            "Central",
            "Outer",
            "Inner",
            "Colonial",
            "Frontier",
            "Gateway",
        ]

        self.name_stems = [
            "Centauri",
            "Orionis",
            "Lyrae",
            "Draconis",
            "Pegasi",
            "Andromedae",
            "Cygni",
            "Aquilae",
            "Bootis",
            "Cassiopeiae",
            "Cephei",
            "Geminorum",
            "Leonis",
            "Persei",
            "Sagittarii",
            "Scorpii",
            "Tauri",
            "Ursae",
            "Virginis",
            "Vulpeculae",
        ]

        # Fictional system names
        self.system_names = [
            "Helios",
            "Aether",
            "Nova",
            "Stellar",
            "Cosmos",
            "Galaxia",
            "Nebula",
            "Quasar",
            "Pulsar",
            "Vortex",
            "Zenith",
            "Apex",
            "Horizon",
            "Eclipse",
            "Solara",
            "Lumina",
            "Celestia",
            "Astral",
        ]

    def generate_body_name(self) -> str:
        """
        Generate a plausible fictional celestial body name.
        """
        name_type = random.choice(["simple", "compound", "greek"])

        if name_type == "simple":
            prefix = random.choice(self.name_prefixes)
            stem = random.choice(self.name_stems)
            return f"{prefix} {stem}"

        elif name_type == "compound":
            prefix = random.choice(self.name_prefixes)
            suffix = random.choice(self.name_suffixes)
            stem = random.choice(self.name_stems)
            return f"{prefix} {stem} {suffix}"

        else:  # greek style
            greek_letters = [
                "Alpha",
                "Beta",
                "Gamma",
                "Delta",
                "Epsilon",
                "Zeta",
            ]
            letter = random.choice(greek_letters)
            stem = random.choice(self.name_stems)
            return f"{letter} {stem}"

    def generate_system_name(self) -> str:
        """
        Generate a fictional solar system name.
        """
        system_type = random.choice(["standard", "numbered", "descriptive"])

        if system_type == "standard":
            return f"{random.choice(self.system_names)} System"
        elif system_type == "numbered":
            return f"{random.choice(self.system_names)} {random.randint(1, 999)}"
        else:
            descriptors = [
                "Binary",
                "Trinary",
                "Quaternary",
                "Complex",
                "Ancient",
                "Young",
            ]
            return f"{random.choice(descriptors)} {random.choice(self.system_names)} System"

    def generate_epoch_time(self) -> str:
        """
        Generate a realistic epoch time within a reasonable range.
        """
        # Generate a random date between 2000 and 2050
        start_date = datetime(2000, 1, 1)
        end_date = datetime(2050, 12, 31)

        time_between = end_date - start_date
        random_days = random.randrange(time_between.days)
        random_date = start_date + timedelta(days=random_days)

        # Add random time
        random_hours = random.randint(0, 23)
        random_minutes = random.randint(0, 59)
        random_seconds = random.randint(0, 59)

        epoch_time = random_date.replace(
            hour=random_hours, minute=random_minutes, second=random_seconds
        )
        return epoch_time.strftime("%Y-%m-%d %H:%M:%S")

    def generate_gravitational_parameter(self) -> float:
        """
        Generate a realistic gravitational parameter (mu) in m³/s².
        """
        # Use log-uniform distribution for more realistic spread
        log_mu_min = math.log10(self.mu_range[0])
        log_mu_max = math.log10(self.mu_range[1])
        log_mu = random.uniform(log_mu_min, log_mu_max)
        return 10**log_mu

    def generate_orbital_position(self, distance_scale: float = 1.0) -> List[float]:
        """
        Generate a random orbital position vector in meters.
        Uses spherical coordinates for more realistic distribution.
        """
        # Distance from center (tend toward smaller distances)
        r = (
            random.uniform(self.orbit_radius_range[0], self.orbit_radius_range[1])
            * distance_scale
        )

        # Spherical coordinates
        theta = random.uniform(0, 2 * math.pi)  # azimuthal angle
        phi = random.uniform(0, math.pi)  # polar angle

        # Convert to Cartesian coordinates
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)

        return [x, y, z]

    def generate_orbital_velocity(
        self, position: List[float], central_mu: float = 1.327e20
    ) -> List[float]:
        """
        Generate a physically plausible orbital velocity vector in m/s.
        Based on circular orbit approximation.
        """
        r = math.sqrt(position[0] ** 2 + position[1] ** 2 + position[2] ** 2)

        if r == 0:
            return [0.0, 0.0, 0.0]

        # Circular orbital velocity: v = sqrt(mu / r)
        circular_velocity = math.sqrt(central_mu / r)

        # Add some random variation (±30%)
        velocity_magnitude = circular_velocity * random.uniform(0.7, 1.3)

        # Generate random direction perpendicular to position vector (orbital plane)
        # Find a vector perpendicular to position
        if abs(position[0]) > 1e-6 or abs(position[1]) > 1e-6:
            temp_vector = [-position[1], position[0], 0]
        else:
            temp_vector = [0, -position[2], position[1]]

        # Normalize temp vector
        temp_mag = math.sqrt(sum(x**2 for x in temp_vector))
        if temp_mag > 0:
            temp_vector = [x / temp_mag for x in temp_vector]

        # Velocity is perpendicular to position (tangential)
        velocity_direction = [
            temp_vector[1] * position[2] - temp_vector[2] * position[1],
            temp_vector[2] * position[0] - temp_vector[0] * position[2],
            temp_vector[0] * position[1] - temp_vector[1] * position[0],
        ]

        # Normalize and scale
        vel_mag = math.sqrt(sum(x**2 for x in velocity_direction))
        if vel_mag > 0:
            velocity = [x * velocity_magnitude / vel_mag for x in velocity_direction]
        else:
            velocity = [velocity_magnitude, 0, 0]  # fallback

        # Add some random inclination variation
        inclination = random.uniform(-0.3, 0.3)
        velocity[2] += velocity_magnitude * inclination

        return velocity

    def generate_single_body(
        self,
        body_id: int,
        central_mu: float = 1.327e20,
        distance_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Generate data for a single fictional celestial body.
        """
        name = self.generate_body_name()
        mu = self.generate_gravitational_parameter()
        r_0 = self.generate_orbital_position(distance_scale)
        v_0 = self.generate_orbital_velocity(r_0, central_mu)

        # Calculate mass from gravitational parameter (mu = G * M)
        G = 6.67430e-11  # gravitational constant
        mass = mu / G if G > 0 else 0

        return {
            "name": name,
            "id": body_id,
            "mu": mu,
            "mass": mass,
            "r_0": r_0,
            "v_0": v_0,
            "radius": self.estimate_body_radius(mass),  # Estimated physical radius
            "color": self.generate_body_color(mass),  # Visual representation color
        }

    def estimate_body_radius(self, mass: float) -> float:
        """
        Estimate body radius based on mass using reasonable density assumptions.
        """
        # Assume average density of 2000 kg/m³ (between gas giants and rocky planets)
        avg_density = 2000  # kg/m³
        volume = mass / avg_density
        radius = (3 * volume / (4 * math.pi)) ** (1 / 3)
        return max(radius, 1e5)  # Minimum 100km radius

    def generate_body_color(self, mass: float) -> List[float]:
        """
        Generate a RGB color for the body based on its mass.
        """
        # Normalize mass to 0-1 range (log scale)
        log_mass_min = math.log10(self.body_mass_range[0])
        log_mass_max = math.log10(self.body_mass_range[1])
        log_mass = math.log10(max(mass, self.body_mass_range[0]))
        normalized_mass = (log_mass - log_mass_min) / (log_mass_max - log_mass_min)

        # Color gradient from blue (small) to red (large)
        if normalized_mass < 0.33:
            # Blue to green
            r = 0.2
            g = normalized_mass * 3
            b = 1.0 - normalized_mass
        elif normalized_mass < 0.66:
            # Green to yellow
            r = (normalized_mass - 0.33) * 3
            g = 1.0
            b = 0.2
        else:
            # Yellow to red
            r = 1.0
            g = 1.0 - (normalized_mass - 0.66) * 3
            b = 0.2

        return [r, g, b]

    def generate_metadata(
        self,
        num_bodies: int,
        epoch: Optional[str] = None,
        center_body: str = "@0",
        system_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate realistic metadata matching Horizons API format.
        """
        if epoch is None:
            epoch = self.generate_epoch_time()

        if system_name is None:
            system_name = self.generate_system_name()

        # Convert epoch to Julian Date for filename compatibility
        epoch_dt = datetime.strptime(epoch, "%Y-%m-%d %H:%M:%S")
        jd = datetime_to_jd(epoch_dt)

        return {
            "generated_utc": datetime.now(UTC).isoformat(),
            "center_body": center_body,
            "epoch": epoch,
            "epoch_jd": jd,
            "target_count": num_bodies,
            "system_name": system_name,
            "data_type": "fictional",
            "description": f"Fictional solar system data for {system_name}",
            "simulation_ready": True,
            "coordinate_frame": "ICRF",
            "distance_units": "meters",
            "velocity_units": "m/s",
            "mu_units": "m³/s²",
        }

    def generate_solar_system(
        self,
        num_bodies: int = 10,
        central_body_mu: float = 1.32712440041279419e20,  # Sun's mu
        distance_scale: float = 1.0,
        include_central_body: bool = True,
        epoch: Optional[str] = None,
        center_body: str = "@0",
        system_name: Optional[str] = None,
        seed: int = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete fictional solar system with proper metadata.
        """
        if seed is not None:
            random.seed(seed)

        print("🌌 GENERATING FICTIONAL SOLAR SYSTEM")
        print(f"🎯 Number of bodies: {num_bodies}")
        print(f"📏 Distance scale: {distance_scale}")
        print(f"⭐ Include central body: {include_central_body}")
        print(f"🕐 Epoch: {epoch if epoch else 'Auto-generated'}")
        print(f"📌 Center: {center_body}")

        body_list = []
        body_id = 0

        # Add central star if requested
        if include_central_body:
            central_body_data = {
                "name": "Helios Prime",
                "mu": central_body_mu,
                "r_0": [0.0, 0.0, 0.0],
                "v_0": [0.0, 0.0, 0.0],
                "mass": central_body_mu / 6.67430e-11,
                "radius": 6.96e8,  # Similar to Sun
                "color": [1.0, 0.9, 0.1],  # Yellow
                "is_central": True,
            }
            body_list.append(central_body_data)
            body_id += 1
            print(f"✅ Added central body: {central_body_data['name']}")

        # Generate orbiting bodies
        for i in range(num_bodies):
            body = self.generate_single_body(body_id, central_body_mu, distance_scale)
            # Remove ID from final output to match Horizons format
            body.pop("id", None)
            body_list.append(body)
            body_id += 1
            print(f"✅ Generated body {i + 1}/{num_bodies}: {body['name']}")

        # Generate metadata
        metadata = self.generate_metadata(
            num_bodies=len(body_list),
            epoch=epoch,
            center_body=center_body,
            system_name=system_name,
        )

        result_data = {"metadata": metadata, "body_list": body_list}

        print(f"🎉 Successfully generated {len(body_list)} bodies")
        print(
            f"📊 Metadata: {system_name if system_name else 'Auto-named system'} at epoch {metadata['epoch']}"
        )

        return result_data

    def save_fictional_data(
        self,
        data: Dict[str, Any],
        filename: str = "fictional_solar_system",
        data_dir: pathlib.Path = None,
        use_epoch_in_filename: bool = True,
    ) -> pathlib.Path:
        """
        Save fictional solar system data to JSON file with proper naming.
        """
        if data_dir is None:
            data_dir = Dir.data

        # Use epoch in filename if requested and available
        if (
            use_epoch_in_filename
            and "metadata" in data
            and "epoch_jd" in data["metadata"]
        ):
            jd = data["metadata"]["epoch_jd"]
            filename = f"{filename}_{jd:.0f}.json"
        elif not filename.endswith(".json"):
            filename += ".json"

        file_path = data_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved fictional data to: {file_path}")
        print(f"📊 File size: {file_path.stat().st_size / 1024:.2f} KB")

        return file_path


def create_fictional_solar_system(
    num_bodies: int = 10,
    central_body_mu: float = 1.32712440041279419e20,
    distance_scale: float = 1.0,
    include_central_body: bool = True,
    epoch: Optional[str] = None,
    center_body: str = "@0",
    system_name: Optional[str] = None,
    seed: int = None,
    filename: str = "fictional_solar_system",
    save: bool = True,
    use_epoch_in_filename: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to create and optionally save a fictional solar system.
    """
    generator = FictionalSolarSystemGenerator(seed=seed)
    system_data = generator.generate_solar_system(
        num_bodies=num_bodies,
        central_body_mu=central_body_mu,
        distance_scale=distance_scale,
        include_central_body=include_central_body,
        epoch=epoch,
        center_body=center_body,
        system_name=system_name,
        seed=seed,
    )

    if save:
        file_path = generator.save_fictional_data(
            system_data, filename, use_epoch_in_filename=use_epoch_in_filename
        )
        system_data["file_path"] = str(file_path)

    return system_data


def load_fictional_data(filename: str, data_dir: pathlib.Path = None) -> Dict[str, Any]:
    """
    Load fictional solar system data from JSON file.
    """
    if data_dir is None:
        data_dir = Dir.data

    if not filename.endswith(".json"):
        filename += ".json"

    file_path = data_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Fictional data file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📂 Loaded fictional data from: {file_path.name}")
    return data


def create_fictional_horizons_like_data(
    body_names: List[str] = None,
    epoch: str = "2025-10-17 12:00:00",
    center_body: str = "@0",
    filename: str = "fictional_horizons_data",
) -> Dict[str, Any]:
    """
    Create fictional data that exactly matches Horizons API structure for specific bodies.
    """
    if body_names is None:
        body_names = ["Helios Prime", "Terra Nova", "Luna Secunda"]

    generator = FictionalSolarSystemGenerator(seed=42)

    # Generate realistic mu values based on body types
    mu_values = []
    for name in body_names:
        if "sun" in name.lower() or "helios" in name.lower():
            mu_values.append(random.uniform(1e20, 5e20))  # Star-like
        elif "terra" in name.lower() or "earth" in name.lower():
            mu_values.append(random.uniform(3e14, 5e14))  # Earth-like
        elif "luna" in name.lower() or "moon" in name.lower():
            mu_values.append(random.uniform(4e12, 8e12))  # Moon-like
        else:
            mu_values.append(generator.generate_gravitational_parameter())

    body_list = []
    central_mu = (
        mu_values[0]
        if "sun" in body_names[0].lower() or "helios" in body_names[0].lower()
        else 1.32712440041279419e20
    )

    for i, (name, mu) in enumerate(zip(body_names, mu_values)):
        if i == 0:  # Central body
            body_data = {
                "name": name,
                "mu": mu,
                "r_0": [0.0, 0.0, 0.0],
                "v_0": [0.0, 0.0, 0.0],
            }
        else:
            position = generator.generate_orbital_position()
            velocity = generator.generate_orbital_velocity(position, central_mu)
            body_data = {
                "name": name,
                "mu": mu,
                "r_0": position,
                "v_0": velocity,
            }
        body_list.append(body_data)

    metadata = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "center_body": center_body,
        "epoch": epoch,
        "target_count": len(body_list),
        "data_type": "fictional_horizons_like",
    }

    result_data = {"metadata": metadata, "body_list": body_list}

    if filename:
        generator.save_fictional_data(
            result_data, filename, use_epoch_in_filename=False
        )

    return result_data


if __name__ == "__main__":
    print("🚀 FICTIONAL SOLAR SYSTEM GENERATOR")
    print("=" * 50)

    # Generate a custom fictional solar system
    print("\n📋 EXAMPLE 2: Custom fictional solar system")
    system = create_fictional_solar_system(
        epoch="2025-10-10 12:00:00",
        num_bodies=25,
        filename="fictional_system",
    )

    # Print summary
    metadata = system["metadata"]
    bodies = system["body_list"]

    print("\n📊 GENERATION SUMMARY:")
    print(f"   System: {metadata.get('system_name', 'Unknown')}")
    print(f"   Total bodies: {len(bodies)}")
    print(f"   Epoch: {metadata['epoch']}")
    print(f"   Center: {metadata['center_body']}")
