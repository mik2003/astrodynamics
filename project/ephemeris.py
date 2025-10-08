import json
import time
from typing import Optional

import requests


def horizons_request(
    format: str = "json",
    command: Optional[str] = None,
    obj_data: bool = True,
    make_ephem: bool = True,
    ephem_type: str = "VECTORS",
    email_addr: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> requests.Response:
    """
    Make a request to NASA Horizons API with comprehensive logging and retry logic.
    """

    print("🚀 INITIATING NASA HORIZONS API REQUEST")
    print("=" * 50)

    # Build the request URL and parameters
    base_url = "https://ssd.jpl.nasa.gov/api/horizons.api"
    params = {
        "format": format,
        "COMMAND": command,
        "OBJ_DATA": "YES" if obj_data else "NO",
        "MAKE_EPHEM": "YES" if make_ephem else "NO",
        "EPHEM_TYPE": ephem_type,
    }

    if email_addr:
        params["EMAIL_ADDR"] = email_addr

    # Remove None values
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
            # Show request details
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

            # Check HTTP status
            response.raise_for_status()

            # Check for API-specific rate limiting messages
            response_lower = response.text.lower()
            if any(
                keyword in response_lower
                for keyword in ["exceeded", "rate limit", "too many", "busy"]
            ):
                print("⚠️  API RATE LIMIT WARNING DETECTED IN RESPONSE")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (
                        2**attempt
                    )  # Exponential backoff
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
            if e.response.status_code == 429:  # Too Many Requests
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
            # Check for common Horizons API response fields
            if "signature" in parsed_data:
                print(
                    f"   API Signature: {parsed_data['signature'].get('version', 'N/A')}"
                )
            if "result" in parsed_data:
                result_preview = (
                    str(parsed_data["result"])[:100] + "..."
                    if len(str(parsed_data["result"])) > 100
                    else str(parsed_data["result"])
                )
                print(f"   Result preview: {result_preview}")

        return parsed_data

    except json.JSONDecodeError as e:
        print(f"❌ JSON PARSE FAILED: {e}")
        print(f"📄 Response preview (first 500 chars):")
        print(response_text[:500])
        raise e


# Main execution with comprehensive logging
print("🌍 NASA HORIZONS API CLIENT")
print("=" * 50)

try:
    # Make the API request
    r = horizons_request(
        command="499",  # Mars
        email_addr="your_email@example.com",  # Replace with your email
    )

    print("\n" + "=" * 50)
    print("📦 PROCESSING SUCCESSFUL RESPONSE")
    print("=" * 50)

    # Parse and display the JSON response
    jr = safe_json_parse(r.text)

    print("\n🎉 FINAL RESULTS")
    print("=" * 50)
    print(json.dumps(jr, indent=2))

except Exception as e:
    print("\n💥 EXECUTION FAILED")
    print("=" * 50)
    print(f"Error: {e}")
    print("\n🔧 TROUBLESHOOTING SUGGESTIONS:")
    print("   1. Check your VPN connection")
    print("   2. Verify your email address is valid")
    print("   3. Try a different target (e.g., '399' for Earth)")
    print("   4. Check NASA Horizons status page")
