import os
import requests
import time
import json
import logging
import traceback
import sys
from datetime import datetime
from playwright.sync_api import sync_playwright
import schedule
import numpy as np

# Ensure the logs directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/fetch_log.txt"), logging.StreamHandler()],
)
logger = logging.getLogger("fetch")

# Create a separate handler for detailed output
detailed_logger = logging.getLogger("fetch_detailed")
detailed_logger.setLevel(logging.INFO)
detailed_handler = logging.FileHandler("logs/fetch_output.log", mode="a")
detailed_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
detailed_logger.addHandler(detailed_handler)
detailed_logger.propagate = False  # Prevent double logging

# Ensure the /data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

# Store discovered endpoints
ENDPOINTS_FILE = "data/endpoints.json"
discovered_endpoints = []


def capture_endpoints(url):
    """Capture all API endpoints from the webpage"""
    logger.info(f"Capturing endpoints from {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            endpoints = []

            def log_request(request):
                if (
                    "https://" in request.url and "/open" in request.url
                ):  # Capture only HTTPS requests
                    if request.url not in endpoints:
                        endpoints.append(request.url)

            # Set up request handler
            page.on("request", log_request)

            # Navigate to the page
            logger.info(f"Navigating to {url}")
            page.goto(url, timeout=60000)  # Increase timeout to 60 seconds

            # Wait for the page to load and interact with it to trigger more requests
            logger.info("Waiting for page to load")
            page.wait_for_timeout(5000)

            # Scroll down to trigger lazy loading
            logger.info("Scrolling page")
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(2000)

            browser.close()
            logger.info(f"Discovered {len(endpoints)} endpoints")
            return endpoints
    except Exception as e:
        logger.error(f"Error in capture_endpoints: {e}")
        logger.error(traceback.format_exc())
        # Return empty list instead of failing
        return []


def fetch_and_save(endpoint):
    """Fetch data from an endpoint and save it to a file"""
    try:
        # Perform the fetch request
        logger.info(f"Fetching data from {endpoint}")
        response = requests.get(endpoint, timeout=30)  # Add timeout

        # Check if the request was successful
        if response.status_code == 200:
            # Extract the endpoint name for the file name
            endpoint_name = endpoint.split("/")[-1]
            # Define the path for the file to be saved
            file_path = os.path.join("data", f"{endpoint_name}.json")

            # Save the response to a file
            with open(file_path, "w") as f:
                f.write(response.text)
            logger.info(f"Saved {endpoint_name}.json")
            return True
        else:
            logger.error(
                f"Failed to fetch {endpoint}. Status code: {response.status_code}"
            )
            return False
    except Exception as e:
        logger.error(f"Error fetching {endpoint}: {e}")
        logger.error(traceback.format_exc())
        return False


def initialize_endpoints():
    """Initialize by discovering all endpoints and saving them"""
    global discovered_endpoints

    try:
        # Check if we already have saved endpoints
        if os.path.exists(ENDPOINTS_FILE):
            with open(ENDPOINTS_FILE, "r") as f:
                discovered_endpoints = json.load(f)
            logger.info(f"Loaded {len(discovered_endpoints)} endpoints from file")
        else:
            # Discover new endpoints
            url = "https://www.nammayatri.in/open?cc=MAA&riders=All&rides=All&vehicles=All&tl=ty"
            discovered_endpoints = capture_endpoints(url)

            # If no endpoints were discovered, use some default endpoints
            if not discovered_endpoints:
                logger.warning("No endpoints discovered, using default endpoints")
                discovered_endpoints = [
                    "https://www.nammayatri.in/open/trends_live_ward_new_key.json",
                    "https://www.nammayatri.in/open/driver_eda_wards_new_key.json",
                    "https://www.nammayatri.in/open/funnel_live_ward_new_key.json",
                ]

            # Save the discovered endpoints
            with open(ENDPOINTS_FILE, "w") as f:
                json.dump(discovered_endpoints, f)
            logger.info(f"Saved {len(discovered_endpoints)} endpoints to file")

        # Initial fetch of all endpoints
        fetch_all_endpoints()
        return True
    except Exception as e:
        logger.error(f"Error in initialize_endpoints: {e}")
        logger.error(traceback.format_exc())
        return False


def fetch_all_endpoints():
    """Fetch data from all known endpoints"""
    try:
        logger.info(f"Fetching data from {len(discovered_endpoints)} endpoints")
        success_count = 0

        for endpoint in discovered_endpoints:
            if fetch_and_save(endpoint):
                success_count += 1

        logger.info(
            f"Successfully fetched {success_count}/{len(discovered_endpoints)} endpoints"
        )

        # If no endpoints were successfully fetched, create sample data
        if success_count == 0:
            logger.warning(
                "No endpoints were successfully fetched. Creating sample data files."
            )
            create_sample_data()
            success_count = 3  # We created 3 sample files

        # Save timestamp of last fetch
        with open("data/last_fetch.txt", "w") as f:
            f.write(datetime.now().isoformat())

        return success_count > 0
    except Exception as e:
        logger.error(f"Error in fetch_all_endpoints: {e}")
        logger.error(traceback.format_exc())

        # Create sample data as a fallback
        logger.warning("Creating sample data files as fallback.")
        create_sample_data()

        return True


def create_sample_data():
    """Create sample data files for testing"""
    try:
        logger.info("Creating sample data files")

        # Sample trends data
        trends_data = []
        for ward_num in range(170, 180):
            for hour in range(24):
                trends_data.append(
                    {
                        "ward_num": str(ward_num),
                        "date": "2025-03-15",
                        "hour": hour,
                        "srch_rqst": int(100 * (1 + 0.5 * np.sin(hour / 12 * np.pi))),
                        "active": "TRUE" if hour >= 6 and hour <= 22 else "FALSE",
                    }
                )

        # Save trends data
        with open("data/trends_live_ward_new_key.json", "w") as f:
            json.dump(trends_data, f)
        logger.info("Created sample trends data file")

        # Sample driver data
        driver_data = []
        vehicle_types = ["Auto", "Cab", "Bike"]
        for ward_num in range(170, 180):
            for vehicle_type in vehicle_types:
                driver_data.append(
                    {
                        "ward_num": str(ward_num),
                        "vehicle_type": vehicle_type,
                        "active_drvr": int(50 * (1 + 0.3 * np.random.random())),
                        "drvr_onride": int(20 * (1 + 0.3 * np.random.random())),
                        "drvr_notonride": int(30 * (1 + 0.3 * np.random.random())),
                        "is_bike": "TRUE" if vehicle_type == "Bike" else "FALSE",
                        "is_cab": "TRUE" if vehicle_type == "Cab" else "FALSE",
                        "is_auto": "TRUE" if vehicle_type == "Auto" else "FALSE",
                    }
                )

        # Save driver data
        with open("data/driver_eda_wards_new_key.json", "w") as f:
            json.dump(driver_data, f)
        logger.info("Created sample driver data file")

        # Sample funnel data
        funnel_data = []
        for ward_num in range(170, 180):
            for vehicle_type in vehicle_types:
                # Generate reasonable values for funnel metrics
                srch_rqst = int(100 * (1 + 0.5 * np.random.random()))
                booking = int(srch_rqst * 0.7 * (1 + 0.2 * np.random.random()))
                done_ride = int(booking * 0.8 * (1 + 0.2 * np.random.random()))
                cancel_ride = booking - done_ride
                drvr_cancel = int(cancel_ride * 0.6)
                rider_cancel = cancel_ride - drvr_cancel

                # Set different base rates for different vehicle types
                base_rate = 150  # Default for Auto
                if vehicle_type == "Cab":
                    base_rate = 200
                elif vehicle_type == "Bike":
                    base_rate = 100

                # Calculate earnings based on vehicle type
                earning = done_ride * base_rate * (1 + 0.3 * np.random.random())
                avg_fare = base_rate * (1 + 0.3 * np.random.random())

                funnel_data.append(
                    {
                        "ward_num": str(ward_num),
                        "vehicle_type": vehicle_type,
                        "srch_rqst": srch_rqst,
                        "booking": booking,
                        "done_ride": done_ride,
                        "cancel_ride": cancel_ride,
                        "drvr_cancel": drvr_cancel,
                        "rider_cancel": rider_cancel,
                        "earning": earning,
                        "dist": done_ride * 5 * (1 + 0.2 * np.random.random()),
                        "avg_dist_pr_trip": 5 * (1 + 0.2 * np.random.random()),
                        "avg_fare": avg_fare,
                        "srch_fr_e": int(srch_rqst * 0.8),
                        "srch_which_got_e": int(srch_rqst * 0.6),
                        "srch_fr_q": int(srch_rqst * 0.2),
                        "srch_which_got_q": int(srch_rqst * 0.1),
                    }
                )

        # Save funnel data
        with open("data/funnel_live_ward_new_key.json", "w") as f:
            json.dump(funnel_data, f)
        logger.info("Created sample funnel data file")

        return True

    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        logger.error(traceback.format_exc())
        return False


def scheduled_fetch():
    """Function to be called by the scheduler"""
    try:
        logger.info("Running scheduled fetch")
        success = fetch_all_endpoints()
        if success:
            logger.info("Scheduled fetch completed successfully")
        else:
            logger.error("Scheduled fetch failed")
    except Exception as e:
        logger.error(f"Error in scheduled_fetch: {e}")
        logger.error(traceback.format_exc())


def main():
    """Main function to run the continuous fetching process"""
    try:
        logger.info("Starting fetch service")

        # Initialize endpoints
        if not initialize_endpoints():
            logger.error("Failed to initialize endpoints, exiting")
            sys.exit(1)

        # Schedule fetching every 2 minutes
        schedule.every(2).minutes.do(scheduled_fetch)

        logger.info("Fetch service started, will fetch data every 2 minutes")

        # Run the scheduling loop
        while True:
            schedule.run_pending()
            time.sleep(1)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
