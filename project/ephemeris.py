import json
import pathlib
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

from project.utilities import Dir, datetime_to_jd

# NASA Horizons body IDs and names mapping
# Add these to your HORIZONS_BODIES dictionary
HORIZONS_BODIES = {
    "Sun": "10",
    "Mercury": "199",
    "Venus": "299",
    "Earth": "399",
    "Moon": "301",
    "Mars": "499",
    # Jupiter system
    "Jupiter": "599",
    "Io": "501",
    "Europa": "502",
    "Ganymede": "503",
    "Callisto": "504",
    # Saturn system
    "Saturn": "699",
    "Mimas": "601",
    "Enceladus": "602",
    "Tethys": "603",
    "Dione": "604",
    "Rhea": "605",
    "Titan": "606",
    "Iapetus": "608",
    # Uranus system
    "Uranus": "799",
    "Miranda": "705",
    "Ariel": "701",
    "Umbriel": "702",
    "Titania": "703",
    "Oberon": "704",
    # Neptune system
    "Neptune": "899",
    "Triton": "801",
    "Proteus": "808",
    # Pluto system
    "Pluto": "999",
    "Charon": "901",
    # Dwarf planets
    "Ceres": "2000001",
    "Eris": "136199",
    "Haumea": "136108",
    "Makemake": "136472",
}

# Standard gravitational parameters (m^3/s^2) for validation
# Add these to your STANDARD_MU dictionary
STANDARD_MU = {
    "Sun": 1.32712440041279419e20,
    "Mercury": 2.203209e13,
    "Venus": 3.248585926e14,
    "Earth": 3.98600435507e14,
    "Moon": 4.902800e12,
    "Mars": 4.2828372e13,
    # Jupiter system
    "Jupiter": 1.266865349e17,
    "Io": 5.959e12,
    "Europa": 3.202e12,
    "Ganymede": 9.887e12,
    "Callisto": 7.179e12,
    # Saturn system
    "Saturn": 3.79311879e16,
    "Mimas": 2.503e9,
    "Enceladus": 7.207e9,
    "Tethys": 4.121e10,
    "Dione": 7.311e10,
    "Rhea": 1.538e11,
    "Titan": 8.978e11,
    "Iapetus": 1.205e11,
    # Uranus system
    "Uranus": 5.7939399e15,
    "Miranda": 4.319e9,
    "Ariel": 8.346e10,
    "Umbriel": 8.509e10,
    "Titania": 2.269e11,
    "Oberon": 2.053e11,
    # Neptune system
    "Neptune": 6.8365299e15,
    "Triton": 1.428e11,
    "Proteus": 3.357e10,
    # Pluto system
    "Pluto": 8.71e11,
    "Charon": 1.058e11,
    # Dwarf planets
    "Ceres": 6.263e10,
    "Eris": 1.108e12,
    "Haumea": 4.006e11,
    "Makemake": 3.1e11,
}


def horizons_request(
    format_: str = "json",
    command: Optional[str] = None,
    obj_data: bool = True,
    make_ephem: bool = True,
    ephem_type: str = "VECTORS",
    center: str = "@0",  # Solar System Barycenter
    email_addr: Optional[str] = None,
    start_time: Optional[str] = None,  # ADD THIS
    stop_time: Optional[str] = None,  # ADD THIS
    step_size: str = "1d",  # ADD THIS
    max_retries: int = 3,
    retry_delay: int = 5,
) -> requests.Response:
    """
    Make a request to NASA Horizons API with comprehensive logging and retry logic.
    """
    print("🚀 INITIATING NASA HORIZONS API REQUEST")
    print("=" * 50)

    base_url = "https://ssd.jpl.nasa.gov/api/horizons.api"
    params = {
        "format": format_,
        "COMMAND": command,
        "OBJ_DATA": "YES" if obj_data else "NO",
        "MAKE_EPHEM": "YES" if make_ephem else "NO",
        "EPHEM_TYPE": ephem_type,
        "CENTER": center,
    }

    if email_addr:
        params["EMAIL_ADDR"] = email_addr

    # ADD THESE TIME PARAMETERS
    if start_time:
        params["START_TIME"] = (
            f"'{start_time}'"  # Horizons expects quoted times
        )
    if stop_time:
        params["STOP_TIME"] = f"'{stop_time}'"
    if step_size:
        params["STEP_SIZE"] = f"'{step_size}'"

    params = {k: v for k, v in params.items() if v is not None}

    print("📋 REQUEST PARAMETERS:")
    for key, value in params.items():
        print(f"   {key}: {value}")

    print(
        f"\n🔄 RETRY CONFIG: {max_retries} attempts with {retry_delay}s initial delay"
    )

    for attempt in range(max_retries):
        print(f"\n🔄 ATTEMPT {attempt + 1}/{max_retries}")
        print("-" * 30)

        try:
            print("📤 SENDING REQUEST...")
            print(f"   URL: {base_url}")
            print(f"   Method: GET")
            print(f"   Timeout: 30 seconds")

            start_time = time.time()
            response = requests.get(base_url, params=params, timeout=30)
            elapsed_time = time.time() - start_time

            print(f"✅ REQUEST COMPLETED")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response Time: {elapsed_time:.2f} seconds")
            print(f"   Content Length: {len(response.text)} bytes")

            response.raise_for_status()

            response_lower = response.text.lower()
            if any(
                keyword in response_lower
                for keyword in ["exceeded", "rate limit", "too many", "busy"]
            ):
                print("⚠️  API RATE LIMIT WARNING DETECTED IN RESPONSE")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print("❌ Max retries reached for rate limiting")
                    return response

            print("🎯 SUCCESS - Valid response received")
            return response

        except requests.exceptions.ConnectionError as e:
            print(f"❌ CONNECTION ERROR: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print("💥 MAX RETRIES REACHED - Connection failed")
                raise Exception(
                    f"Failed after {max_retries} connection attempts: {e}"
                )

        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP ERROR: {e}")
            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    print(
                        f"⏳ HTTP 429 - Rate limited. Waiting {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                    continue
            print(f"📊 Response headers: {dict(e.response.headers)}")
            raise e

        except requests.exceptions.Timeout:
            print("⏰ REQUEST TIMEOUT - No response after 30 seconds")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                raise Exception(
                    f"Request timed out after {max_retries} attempts"
                )

        except requests.exceptions.RequestException as e:
            print(f"❌ REQUEST EXCEPTION: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                raise Exception(
                    f"Request failed after {max_retries} attempts: {e}"
                )

    raise Exception(f"All {max_retries} attempts failed")


def safe_json_parse(response_text: str):
    """
    Safely parse JSON response with error handling and logging
    """
    print("\n📊 PARSING JSON RESPONSE")
    print("-" * 30)

    try:
        print("🔍 Attempting to parse JSON...")
        parsed_data = json.loads(response_text)
        print("✅ JSON PARSE SUCCESSFUL")
        print(f"   Response type: {type(parsed_data)}")

        if isinstance(parsed_data, dict):
            print(f"   Keys found: {list(parsed_data.keys())}")
            if "signature" in parsed_data:
                print(
                    f"   API Signature: {parsed_data['signature'].get('version', 'N/A')}"
                )

            # CHECK FOR ERRORS
            if "error" in parsed_data and parsed_data["error"]:
                print(f"❌ API ERROR DETECTED: {parsed_data['error']}")

            if "result" in parsed_data:
                result_preview = (
                    str(parsed_data["result"])[:200] + "..."
                    if len(str(parsed_data["result"])) > 200
                    else str(parsed_data["result"])
                )
                print(f"   Result preview: {result_preview}")

        return parsed_data

    except json.JSONDecodeError as e:
        print(f"❌ JSON PARSE FAILED: {e}")
        print(f"📄 Response preview (first 500 chars):")
        print(response_text[:500])
        raise e


def extract_gravitational_parameter(result_text: str) -> float:
    """
    Extract GM (gravitational parameter) from Horizons result text.
    Returns value in m^3/s^2.
    """
    # Look for GM pattern in the result text
    gm_patterns = [
        r"GM \(km\^3/s\^2\)\s*=\s*([\d.+-E]+)",
        r"GM,\s*km\^3/s\^2\s*:\s*([\d.+-E]+)",
        r"GM\s*=\s*([\d.+-E]+)\s*km\^3/s\^2",
    ]

    for pattern in gm_patterns:
        match = re.search(pattern, result_text)
        if match:
            gm_km3_s2 = float(match.group(1))
            # Convert from km^3/s^2 to m^3/s^2
            gm_m3_s2 = gm_km3_s2 * 1e9
            print(
                f"   ✅ Extracted GM: {gm_km3_s2} km³/s² → {gm_m3_s2:.2e} m³/s²"
            )
            return gm_m3_s2

    print("   ⚠️  GM not found in response, using standard value")
    return None


def extract_ephemeris_data(
    result_text: str,
) -> Tuple[List[float], List[float]]:
    """
    Extract position and velocity vectors from Horizons ephemeris data.
    Returns (position_km, velocity_km_s) where position is in km and velocity in km/s.
    """
    print("   🔍 Searching for ephemeris data...")

    # Find the SOE (Start of Ephemeris) section
    soe_match = re.search(r"\$\$SOE(.*?)\$\$EOE", result_text, re.DOTALL)

    if not soe_match:
        print(f"   ❌ No SOE/EOE markers found in response")
        # Check if there's an error message in the result
        error_match = re.search(r"Error:\s*(.*)", result_text)
        if error_match:
            print(f"   ❌ Horizons Error: {error_match.group(1)}")
        else:
            # Print more context about what we actually received
            lines = result_text.split("\n")
            for i, line in enumerate(lines[:10]):  # Show first 10 lines
                print(f"   Line {i}: {line}")
        raise ValueError("No ephemeris data found (SOE/EOE markers missing)")

    ephemeris_text = soe_match.group(1)
    print(f"   ✅ Found ephemeris data section ({len(ephemeris_text)} chars)")

    # Take the first record
    first_record_match = re.search(
        r"X\s*=\s*([\d.E+-]+)\s+Y\s*=\s*([\d.E+-]+)\s+Z\s*=\s*([\d.E+-]+)\s*"
        r"VX\s*=\s*([\d.E+-]+)\s+VY\s*=\s*([\d.E+-]+)\s+VZ\s*=\s*([\d.E+-]+)",
        ephemeris_text,
    )

    if not first_record_match:
        print(f"   ❌ Could not parse vector data from ephemeris")
        print(f"   📄 First 500 chars of ephemeris: {ephemeris_text[:500]}...")
        raise ValueError("Could not parse ephemeris vectors")

    # Extract position (km) and velocity (km/s)
    position_km = [
        float(first_record_match.group(1)),  # X
        float(first_record_match.group(2)),  # Y
        float(first_record_match.group(3)),  # Z
    ]

    velocity_km_s = [
        float(first_record_match.group(4)),  # VX
        float(first_record_match.group(5)),  # VY
        float(first_record_match.group(6)),  # VZ
    ]

    print(
        f"   ✅ Position: [{position_km[0]:.2e}, {position_km[1]:.2e}, {position_km[2]:.2e}] km"
    )
    print(
        f"   ✅ Velocity: [{velocity_km_s[0]:.2e}, {velocity_km_s[1]:.2e}, {velocity_km_s[2]:.2e}] km/s"
    )

    return position_km, velocity_km_s


def fetch_body_data(
    body_name: str,
    center_body: str = "@0",
    epoch: str = None,
    email_addr: str = None,
) -> Dict[str, Any]:
    """
    Fetch data for a single body from Horizons API.
    """
    print(f"\n🪐 FETCHING DATA FOR: {body_name}")
    print("-" * 40)

    # Get Horizons ID
    body_id = HORIZONS_BODIES.get(body_name)
    if not body_id:
        raise ValueError(f"Unknown body: {body_name}")

    print(f"   Horizons ID: {body_id}")
    print(f"   Center: {center_body}")

    # If epoch is provided, use it with a small time span to ensure ephemeris generation
    if epoch:
        start_time = epoch
        # Add 1 hour to stop time to ensure we get at least one ephemeris record
        stop_time = (
            datetime.strptime(epoch, "%Y-%m-%d %H:%M:%S") + timedelta(hours=1)
        ).strftime("%Y-%m-%d %H:%M:%S")
        step_size = "1h"  # 1 hour step to ensure we get our requested time
    else:
        # Use current time if no epoch specified
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = current_time
        stop_time = (datetime.now() + timedelta(hours=1)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        step_size = "1h"

    # Build request parameters
    request_params = {
        "command": body_id,
        "center": center_body,
        "email_addr": email_addr,
        "start_time": start_time,
        "stop_time": stop_time,
        "step_size": step_size,
    }

    # First request to get physical data (GM)
    print("📡 Requesting physical data...")
    response = horizons_request(**request_params)
    raw_data = safe_json_parse(response.text)

    # Check for API errors
    if raw_data.get("error"):
        raise ValueError(f"Horizons API error: {raw_data['error']}")

    # Extract GM
    gm_m3_s2 = extract_gravitational_parameter(raw_data["result"])

    # Use standard GM if not found in response
    if gm_m3_s2 is None:
        gm_m3_s2 = STANDARD_MU.get(body_name)
        if gm_m3_s2:
            print(f"   📋 Using standard GM: {gm_m3_s2:.2e} m³/s²")
        else:
            raise ValueError(f"Could not determine GM for {body_name}")

    # Second request to get ephemeris data
    print("📡 Requesting ephemeris data...")
    response = horizons_request(**request_params)
    raw_data = safe_json_parse(response.text)

    # Check for API errors in ephemeris request
    if raw_data.get("error"):
        raise ValueError(f"Horizons API error: {raw_data['error']}")

    # Extract position and velocity
    position_km, velocity_km_s = extract_ephemeris_data(raw_data["result"])

    # Convert position from km to meters
    position_m = [coord * 1000 for coord in position_km]

    # Convert velocity from km/s to m/s
    velocity_m_s = [vel * 1000 for vel in velocity_km_s]

    print(
        f"   📍 Final position (m): [{position_m[0]:.2e}, {position_m[1]:.2e}, {position_m[2]:.2e}]"
    )
    print(
        f"   🚀 Final velocity (m/s): [{velocity_m_s[0]:.2e}, {velocity_m_s[1]:.2e}, {velocity_m_s[2]:.2e}]"
    )

    return {
        "name": body_name,
        "mu": gm_m3_s2,
        "r_0": position_m,
        "v_0": velocity_m_s,
    }


def create_solar_system_data(
    target_bodies: List[str],
    center_body: str = "@0",
    epoch: str = None,
    output_filename: str = "solar_system_data",
    email_addr: str = None,
    use_existing: bool = True,
) -> Dict[str, Any]:
    """
    Create solar system data file with body positions, velocities, and gravitational parameters.
    """
    print("🌌 SOLAR SYSTEM DATA GENERATOR")
    print("=" * 50)
    print(f"🎯 Targets: {', '.join(target_bodies)}")
    print(f"📌 Center: {center_body}")
    print(f"🕐 Epoch: {epoch if epoch else 'Current time'}")

    # Try to load existing data first
    if use_existing:
        try:
            latest_file = get_latest_horizons_file(output_filename)
            if latest_file:
                print(f"📂 Using existing data file: {latest_file.name}")
                # FIX: Pass the Path object directly to load_horizons_data
                return load_horizons_data(latest_file)
        except FileNotFoundError:
            print("📭 No existing data found, fetching from API...")

    body_list = []

    for body_name in target_bodies:
        try:
            body_data = fetch_body_data(
                body_name, center_body, epoch, email_addr
            )
            body_list.append(body_data)
            print(f"✅ Successfully processed {body_name}")
        except Exception as e:
            print(f"❌ Failed to process {body_name}: {e}")
            import traceback

            print(f"   Detailed error: {traceback.format_exc()}")
            continue

    if not body_list:
        raise Exception(
            "No bodies were successfully processed. Check the errors above."
        )

    result_data = {
        "metadata": {
            "generated_utc": datetime.utcnow().isoformat(),
            "center_body": center_body,
            "epoch": epoch if epoch else "current",
            "target_count": len(body_list),
        },
        "body_list": body_list,
    }

    # Save to file
    file_path = save_horizons_data(result_data, output_filename, epoch=epoch)

    print(f"\n🎉 SOLAR SYSTEM DATA GENERATION COMPLETE")
    print(f"📊 Bodies processed: {len(body_list)}/{len(target_bodies)}")
    print(f"💾 Output file: {file_path}")

    return result_data


def save_horizons_data(
    data: Dict[str, Any],
    filename: str,
    epoch: str = None,
    data_dir: pathlib.Path = None,
    backup_existing: bool = True,
) -> pathlib.Path:
    """
    Save NASA Horizons JSON data to file with proper project structure.
    """
    if data_dir is None:
        data_dir = Dir.data_dir

    if epoch:
        epoch_d = datetime.strptime(epoch, "%Y-%m-%d %H:%M:%S")
        epoch_j = datetime_to_jd(epoch_d)
        filename = f"{filename}_{epoch_j:.0f}.json"
    else:
        filename = f"{filename}.json"

    file_path = data_dir.joinpath(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if backup_existing and file_path.exists():
        backup_path = file_path.with_suffix(".json.backup")
        print(f"📦 Backing up existing file to: {backup_path.name}")
        file_path.rename(backup_path)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"💾 Data saved to: {file_path}")
    print(f"📊 File size: {file_path.stat().st_size / 1024:.2f} KB")

    return file_path


def load_horizons_data(
    filename: str, data_dir: pathlib.Path = None
) -> Dict[str, Any]:
    """
    Load NASA Horizons JSON data from file.
    """
    if data_dir is None:
        data_dir = Dir.data_dir

    # FIX: Check if filename is already a Path object
    if isinstance(filename, pathlib.Path):
        file_path = filename
    else:
        # Ensure .json extension if it's a string
        if not filename.endswith(".json"):
            filename += ".json"
        file_path = data_dir.joinpath(filename)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📂 Data loaded from: {file_path.name}")
    return data


def get_latest_horizons_file(
    base_filename: str, data_dir: pathlib.Path = None
) -> Optional[pathlib.Path]:
    """
    Find the most recent timestamped file for a given base filename.
    """
    if data_dir is None:
        data_dir = Dir.data_dir

    pattern = f"{base_filename}_*.json"
    matching_files = list(data_dir.glob(pattern))

    if not matching_files:
        return None

    # Return the most recently modified file - FIX: return the actual Path object
    latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
    return latest_file


if __name__ == "__main__":
    # Full solar system including all planets and major dwarf planets
    full_solar_system = create_solar_system_data(
        target_bodies=[
            "Sun",
            "Earth",
            "Moon",
        ],
        center_body="@0",  # Solar System Barycenter
        epoch="2025-10-17 12:00:00",
        output_filename="sun_earth_moon",
        email_addr="michelangelosecondo+horizons@gmail.com",  # Replace with your actual email
        use_existing=False,  # Set to True after first run to use cached data
    )
