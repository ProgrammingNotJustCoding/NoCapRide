import os
import json
import logging
import traceback
from datetime import datetime, timedelta

import pandas as pd
import requests
from custom_logger import CustomLogger

logger = CustomLogger("data_manager", "logs/data_manager_log.txt")


class RideDataManager:
    """Class to manage ride data fetching, storage, and retrieval"""

    def __init__(self):
        self.last_fetch_time = None
        self.historical_data = {}

    def fetch_endpoint(self, endpoint, base_url):
        """Fetch data from a specific endpoint"""
        url = f"{base_url}/{endpoint}"
        try:
            logger.info(f"Fetching data from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            file_path = os.path.join("cache", endpoint)
            with open(file_path, "w") as f:
                f.write(response.text)

            data = json.loads(response.text)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hist_file_path = os.path.join("cache/forecasts", f"{timestamp}_{endpoint}")
            with open(hist_file_path, "w") as f:
                f.write(response.text)

            logger.info(f"Successfully fetched and saved {endpoint}")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {endpoint}: {e}")
            logger.error(traceback.format_exc())
            return self._load_latest_data(endpoint)

    def _load_latest_data(self, endpoint):
        """Load the latest data file if available as fallback"""
        file_path = os.path.join("cache", endpoint)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading latest data for {endpoint}: {e}")
        return None

    def fetch_all_endpoints(self, endpoints, base_url):
        """Fetch data from all endpoints"""
        fetched_data = {}
        fetch_success = False

        for endpoint in endpoints:
            data = self.fetch_endpoint(endpoint, base_url)
            if data is not None:
                fetched_data[endpoint] = data
                fetch_success = True

        if fetch_success:
            self.last_fetch_time = datetime.now()
            self.update_historical_dataset(fetched_data)

        return fetched_data

    def update_historical_dataset(self, new_data):
        """Update the historical dataset with new data"""
        try:
            if "trends_live_ward_new_key.json" in new_data:
                self._process_trends_data(
                    new_data["trends_live_ward_new_key.json"], "ward"
                )

            if "trends_live_ca_new_key.json" in new_data:
                self._process_trends_data(new_data["trends_live_ca_new_key.json"], "ca")

            self._save_historical_data()

            logger.info("Historical dataset updated successfully")
        except Exception as e:
            logger.error(f"Error updating historical dataset: {e}")
            logger.error(traceback.format_exc())

    def _process_trends_data(self, data, data_type):
        """Process trends data and add to historical collection"""
        try:
            if f"trends_{data_type}" not in self.historical_data:
                self.historical_data[f"trends_{data_type}"] = []

            if isinstance(data, dict):
                data = [data]

            current_time = datetime.now()
            for item in data:
                
                if "datetime" not in item or not self._is_valid_datetime(
                    item["datetime"]
                ):
                    item["datetime"] = current_time.isoformat()

                item["timestamp"] = current_time.isoformat()
                self.historical_data[f"trends_{data_type}"].append(item)

            logger.info(f"Processed {len(data)} {data_type} trend records")
            logger.debug(f"Processed data for {data_type}: {data}")
        except Exception as e:
            logger.error(f"Error processing trends data: {e}")
            logger.error(traceback.format_exc())

    def _is_valid_datetime(self, datetime_str):
        """Validate if a string is a valid ISO 8601 datetime"""
        try:
            datetime.fromisoformat(datetime_str)
            return True
        except ValueError:
            return False

    def _save_historical_data(self):
        """Save historical data to files"""
        try:
            for data_key, data_value in self.historical_data.items():
                file_path = os.path.join("cache", f"{data_key}.json")
                with open(file_path, "w") as f:
                    json.dump(data_value, f, indent=2)

            logger.info("Historical data saved to files")
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")
            logger.error(traceback.format_exc())

    def load_historical_data(self):
        """Load historical data from files"""
        try:
            self.historical_data = {}

            for data_type in ["ward", "ca"]:
                file_path = os.path.join("cache", f"trends_{data_type}.json")
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        self.historical_data[f"trends_{data_type}"] = json.load(f)

            self._ensure_minimum_data()

            return True
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            logger.error(traceback.format_exc())
            return False

    def _ensure_minimum_data(self):
        """Ensure we have at least some data by generating synthetic data if needed"""
        min_records = 48

        for data_type in ["ward", "ca"]:
            key = f"trends_{data_type}"
            if (
                key not in self.historical_data
                or len(self.historical_data[key]) < min_records
            ):
                self._generate_synthetic_data(data_type)

    def _generate_synthetic_data(self, data_type):
        """Generate synthetic data for training"""
        key = f"trends_{data_type}"

        end_time = datetime.now()
        start_time = end_time - timedelta(days=3)

        synthetic_data = []

        regions = range(1, 6) if data_type == "ward" else range(1, 4)

        for region_id in regions:
            region_name = str(region_id)

            current_time = start_time
            while current_time < end_time:
                synthetic_data.append(
                    {
                        "region": region_name,
                        "timestamp": current_time.isoformat(),
                        "srch_rqst": 0,
                    }
                )
                current_time += timedelta(hours=1)

        if key in self.historical_data:
            self.historical_data[key].extend(synthetic_data)
        else:
            self.historical_data[key] = synthetic_data

        logger.info(
            f"Generated {len(synthetic_data)} synthetic records for {data_type}"
        )
        logger.debug(f"Synthetic data generated for {data_type}: {synthetic_data}")

    def get_historical_trends_data(self, data_type="ward"):
        """Retrieve historical trends data for the specified type."""
        try:
            key = f"trends_{data_type}"
            if key in self.historical_data:
                return pd.DataFrame(self.historical_data[key])
            else:
                logger.warning(f"No historical data found for {data_type}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving historical trends data: {e}")
            return pd.DataFrame()

    def get_nearby_wards(self, ward, max_distance):
        """Find nearby wards within a given distance."""
        try:
            
            ward_coordinates = {
                "101": (13.0827, 80.2707),  
                "102": (13.0820, 80.2710),
                
            }

            if ward not in ward_coordinates:
                logger.error(f"Ward {ward} not found in coordinates mapping")
                return []

            current_coords = ward_coordinates[ward]

            def haversine(coord1, coord2):
                """Calculate the great-circle distance between two points on the Earth."""
                from math import radians, sin, cos, sqrt, atan2

                lat1, lon1 = radians(coord1[0]), radians(coord1[1])
                lat2, lon2 = radians(coord2[0]), radians(coord2[1])

                dlat = lat2 - lat1
                dlon = lon2 - lon1

                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * atan2(sqrt(a), sqrt(1 - a))

                
                R = 6371.0
                return R * c

            nearby_wards = []
            for other_ward, coords in ward_coordinates.items():
                if other_ward != ward:
                    distance = haversine(current_coords, coords)
                    if distance <= max_distance:
                        nearby_wards.append((other_ward, distance))

            
            nearby_wards.sort(key=lambda x: x[1])

            return [ward for ward, _ in nearby_wards]

        except Exception as e:
            logger.error(f"Error finding nearby wards: {e}")
            return []
