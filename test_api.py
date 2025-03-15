#!/usr/bin/env python3
"""
Test script to check if the API is running correctly.
"""

import requests
import sys
import json
import time
from datetime import datetime
import argparse

# Default API base URL
API_BASE_URL = "http://localhost:8888/api"

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_success(message):
    print(f"{GREEN}✓ {message}{RESET}")


def print_error(message):
    print(f"{RED}✗ {message}{RESET}")


def print_warning(message):
    print(f"{YELLOW}! {message}{RESET}")


def print_info(message):
    print(f"{BLUE}ℹ {message}{RESET}")


def test_health(base_url):
    """Test the health endpoint"""
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/health", timeout=5)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print_success(f"Health check passed in {elapsed:.2f}s: {data}")
            return True
        else:
            print_error(
                f"Health check failed with status {response.status_code}: {response.text}"
            )
            return False
    except Exception as e:
        print_error(f"Health check error: {str(e)}")
        return False


def test_regions(base_url, data_type="ward"):
    """Test the regions endpoint with different filters"""
    try:
        # Test with no filter (all regions)
        start_time = time.time()
        response = requests.get(f"{base_url}/regions?type={data_type}", timeout=10)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            all_regions = data.get("regions", [])
            print_success(
                f"Regions endpoint (all) returned {len(all_regions)} regions in {elapsed:.2f}s"
            )

            # Test with simple filter
            response = requests.get(
                f"{base_url}/regions?type={data_type}&filter=simple", timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                simple_regions = data.get("regions", [])
                print_success(
                    f"Regions endpoint (simple) returned {len(simple_regions)} regions"
                )

            # Test with b_ filter
            response = requests.get(
                f"{base_url}/regions?type={data_type}&filter=b", timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                b_regions = data.get("regions", [])
                print_success(
                    f"Regions endpoint (b_) returned {len(b_regions)} regions"
                )

            # Test with w_ filter
            response = requests.get(
                f"{base_url}/regions?type={data_type}&filter=w", timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                w_regions = data.get("regions", [])
                print_success(
                    f"Regions endpoint (w_) returned {len(w_regions)} regions"
                )

            # Verify that the sum of filtered regions equals all regions
            total_filtered = len(simple_regions) + len(b_regions) + len(w_regions)
            if total_filtered == len(all_regions):
                print_success(
                    f"Region filtering validation passed: {total_filtered} = {len(all_regions)}"
                )
            else:
                print_warning(
                    f"Region filtering validation failed: {total_filtered} != {len(all_regions)}"
                )

            return True
        else:
            print_error(
                f"Regions endpoint failed with status {response.status_code}: {response.text}"
            )
            return False
    except Exception as e:
        print_error(f"Regions endpoint error: {str(e)}")
        return False


def test_historical(base_url, data_type="ward", hours=24, region="1"):
    """Test the historical endpoint"""
    try:
        start_time = time.time()
        response = requests.get(
            f"{base_url}/historical?type={data_type}&hours={hours}&region={region}",
            timeout=20,
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            data_points = len(data.get("data", []))
            print_success(
                f"Historical endpoint returned {data_points} data points in {elapsed:.2f}s"
            )
            return True
        else:
            print_error(
                f"Historical endpoint failed with status {response.status_code}: {response.text}"
            )
            return False
    except Exception as e:
        print_error(f"Historical endpoint error: {str(e)}")
        return False


def test_forecast(base_url, data_type="ward", hours=12, region="1"):
    """Test the forecast endpoint"""
    try:
        print_info(
            f"Testing forecast for region {region}, this may take up to 60 seconds..."
        )
        start_time = time.time()
        response = requests.get(
            f"{base_url}/forecast?type={data_type}&hours={hours}&region={region}",
            timeout=60,
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            forecast_points = len(data.get("forecast", []))
            print_success(
                f"Forecast endpoint returned {forecast_points} data points in {elapsed:.2f}s"
            )
            return True
        else:
            print_error(
                f"Forecast endpoint failed with status {response.status_code}: {response.text}"
            )
            return False
    except requests.exceptions.Timeout:
        print_error(f"Forecast endpoint timed out after 60 seconds")
        return False
    except Exception as e:
        print_error(f"Forecast endpoint error: {str(e)}")
        return False


def test_model_info(base_url, data_type="ward"):
    """Test the model info endpoint"""
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/model/info?type={data_type}", timeout=10)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print_success(
                f"Model info endpoint returned data in {elapsed:.2f}s: {data}"
            )
            return True
        else:
            print_error(
                f"Model info endpoint failed with status {response.status_code}: {response.text}"
            )
            return False
    except Exception as e:
        print_error(f"Model info endpoint error: {str(e)}")
        return False


def test_cache_clear(base_url):
    """Test the cache clear endpoint"""
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/cache/clear", timeout=10)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print_success(f"Cache clear endpoint succeeded in {elapsed:.2f}s: {data}")
            return True
        else:
            print_error(
                f"Cache clear endpoint failed with status {response.status_code}: {response.text}"
            )
            return False
    except Exception as e:
        print_error(f"Cache clear endpoint error: {str(e)}")
        return False


def test_string_region_forecast(base_url, data_type="ward", hours=12):
    """Test the forecast endpoint with string-based region identifiers"""
    # Test with a b_ prefixed region
    b_region_success = False
    try:
        print_info("Testing forecast with b_ prefixed region...")
        response = requests.get(
            f"{base_url}/regions?type={data_type}&filter=b", timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            b_regions = data.get("regions", [])
            if b_regions:
                test_region = b_regions[0]
                print_info(
                    f"Testing forecast for region {test_region}, this may take up to 60 seconds..."
                )
                start_time = time.time()
                response = requests.get(
                    f"{base_url}/forecast?type={data_type}&hours={hours}&region={test_region}",
                    timeout=60,
                )
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()
                    forecast_points = len(data.get("forecast", []))
                    print_success(
                        f"Forecast with b_ region returned {forecast_points} data points in {elapsed:.2f}s"
                    )
                    b_region_success = True
                else:
                    print_error(
                        f"Forecast with b_ region failed with status {response.status_code}: {response.text}"
                    )
            else:
                print_warning("No b_ prefixed regions found to test")
    except Exception as e:
        print_error(f"Forecast with b_ region error: {str(e)}")

    # Test with a w_ prefixed region
    w_region_success = False
    try:
        print_info("Testing forecast with w_ prefixed region...")
        response = requests.get(
            f"{base_url}/regions?type={data_type}&filter=w", timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            w_regions = data.get("regions", [])
            if w_regions:
                test_region = w_regions[0]
                print_info(
                    f"Testing forecast for region {test_region}, this may take up to 60 seconds..."
                )
                start_time = time.time()
                response = requests.get(
                    f"{base_url}/forecast?type={data_type}&hours={hours}&region={test_region}",
                    timeout=60,
                )
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()
                    forecast_points = len(data.get("forecast", []))
                    print_success(
                        f"Forecast with w_ region returned {forecast_points} data points in {elapsed:.2f}s"
                    )
                    w_region_success = True
                else:
                    print_error(
                        f"Forecast with w_ region failed with status {response.status_code}: {response.text}"
                    )
            else:
                print_warning("No w_ prefixed regions found to test")
    except Exception as e:
        print_error(f"Forecast with w_ region error: {str(e)}")

    return b_region_success or w_region_success


def test_forecast_all(
    base_url, data_type="ward", hours=12, filter_type="all", workers=4
):
    """Test the forecast/all endpoint that returns forecasts for all regions"""
    try:
        print_info(
            f"Testing forecast/all endpoint with filter={filter_type}, workers={workers}, this may take up to 90 seconds..."
        )
        start_time = time.time()
        response = requests.get(
            f"{base_url}/forecast/all?type={data_type}&hours={hours}&filter={filter_type}&workers={workers}",
            timeout=90,  # Longer timeout since this processes multiple regions
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            forecasts = data.get("forecasts", {})
            region_count = len(forecasts)
            total_forecast_points = sum(len(points) for points in forecasts.values())

            print_success(
                f"Forecast/all endpoint returned forecasts for {region_count} regions "
                f"with {total_forecast_points} total data points in {elapsed:.2f}s "
                f"using {data.get('metadata', {}).get('workers', 'unknown')} workers"
            )

            # Print some sample regions
            if region_count > 0:
                sample_regions = list(forecasts.keys())[:3]  # Show up to 3 regions
                print_info(f"Sample regions: {', '.join(sample_regions)}")

                # Show forecast points for first region
                if sample_regions:
                    first_region = sample_regions[0]
                    first_region_points = len(forecasts[first_region])
                    print_info(
                        f"Region {first_region} has {first_region_points} forecast points"
                    )

            return True
        else:
            print_error(
                f"Forecast/all endpoint failed with status {response.status_code}: {response.text}"
            )
            return False
    except requests.exceptions.Timeout:
        print_error(f"Forecast/all endpoint timed out after 90 seconds")
        return False
    except Exception as e:
        print_error(f"Forecast/all endpoint error: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test the NoCapRide API endpoints")
    parser.add_argument("--url", default=API_BASE_URL, help="Base URL for the API")
    parser.add_argument(
        "--type",
        default="ward",
        choices=["ward", "ca"],
        help="Data type to use for testing",
    )
    parser.add_argument("--region", default="1", help="Region to use for testing")
    parser.add_argument(
        "--hours", default=12, type=int, help="Hours to use for testing"
    )
    parser.add_argument(
        "--skip-forecast",
        action="store_true",
        help="Skip forecast tests (which can be slow)",
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        help="Number of worker threads for parallel forecasting",
    )
    args = parser.parse_args()

    base_url = args.url
    data_type = args.type
    region = args.region
    hours = args.hours
    workers = args.workers

    print_info(f"Testing API at {base_url}")
    print_info(
        f"Using data_type={data_type}, region={region}, hours={hours}, workers={workers}"
    )

    # Run the tests
    tests_passed = 0
    tests_failed = 0

    # Test health endpoint
    if test_health(base_url):
        tests_passed += 1
    else:
        tests_failed += 1

    # Test regions endpoint
    if test_regions(base_url, data_type):
        tests_passed += 1
    else:
        tests_failed += 1

    # Test historical endpoint
    if test_historical(base_url, data_type, hours, region):
        tests_passed += 1
    else:
        tests_failed += 1

    # Test forecast endpoint
    if not args.skip_forecast:
        if test_forecast(base_url, data_type, hours, region):
            tests_passed += 1
        else:
            tests_failed += 1

        # Test string-based region identifiers
        if test_string_region_forecast(base_url, data_type, hours):
            tests_passed += 1
        else:
            tests_failed += 1

        # Test forecast/all endpoint with specified worker count
        if test_forecast_all(base_url, data_type, hours, "all", workers):
            tests_passed += 1
        else:
            tests_failed += 1
    else:
        print_warning("Skipping forecast tests")

    # Test model info endpoint
    if test_model_info(base_url, data_type):
        tests_passed += 1
    else:
        tests_failed += 1

    # Test cache clear endpoint
    if test_cache_clear(base_url):
        tests_passed += 1
    else:
        tests_failed += 1

    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    print("=" * 50)

    # Return non-zero exit code if any tests failed
    if tests_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
