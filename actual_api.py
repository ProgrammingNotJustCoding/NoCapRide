import os
import json
import logging
import traceback
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle
import threading
import functools
from werkzeug.serving import run_simple
import re
import concurrent.futures
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

# Import from the existing forecasting module
from panda import RideDataManager, RideRequestForecast

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api_log.txt", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("forecast_api")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize data manager and forecaster
data_manager = RideDataManager()
forecaster = RideRequestForecast(data_manager)

# Cache for forecast results
forecast_cache = {}
# Lock for thread safety
cache_lock = threading.Lock()

# Ensure cache directory exists
CACHE_DIR = "cache/forecasts"
os.makedirs(CACHE_DIR, exist_ok=True)

# Load historical data and models
try:
    data_manager.load_historical_data()
    forecaster.load_model("ward")
    logger.info("Historical data and models loaded successfully")
except Exception as e:
    logger.error(f"Error loading data or models: {e}")
    logger.error(traceback.format_exc())

# Constants for pricing model
MINIMUM_PRICE = 30  # ₹30 minimum starting fare
PER_KM_VALUE = 15  # ₹15 per kilometer
PER_MIN_CHARGE = 1.5  # ₹1.5 per minute in traffic


def parse_iso_date(date_string):
    """Parse ISO format dates with Z timezone indicator"""
    if date_string.endswith('Z'):
        date_string = date_string[:-1] + '+00:00'
    return datetime.fromisoformat(date_string)

# Cache decorator with timeout
def timed_cache(timeout_seconds=300):
    def decorator(func):
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            current_time = datetime.now()

            with cache_lock:
                # Check if result is in cache and not expired
                if key in cache:
                    result, timestamp = cache[key]
                    if (current_time - timestamp).total_seconds() < timeout_seconds:
                        logger.info(
                            f"Cache hit for {func.__name__} with args {args}, kwargs {kwargs}"
                        )
                        return result

            # Generate new result
            result = func(*args, **kwargs)

            with cache_lock:
                # Store in cache
                cache[key] = (result, current_time)

            return result

        return wrapper

    return decorator


def get_cache_file_path(cache_key):
    """Generate a sanitized file path for the cache key"""
    # Replace characters that aren't valid in filenames
    safe_key = re.sub(r"[^\w\-_]", "_", cache_key)
    return os.path.join(CACHE_DIR, f"{safe_key}.json")


def load_from_file_cache(cache_key, max_age_seconds=1800):
    """Load data from file cache if it exists and is not expired"""
    file_path = get_cache_file_path(cache_key)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                cache_data = json.load(f)

            # Check if the cache is still valid
            cache_time = datetime.fromisoformat(
                cache_data.get("cached_at", "2000-01-01T00:00:00")
            )
            if (datetime.now() - cache_time).total_seconds() < max_age_seconds:
                logger.info(f"Using file cache for {cache_key}")
                return cache_data.get("data")
        except Exception as e:
            logger.error(f"Error loading from file cache: {e}")

    return None


def save_to_file_cache(cache_key, data):
    """Save data to file cache"""
    file_path = get_cache_file_path(cache_key)
    try:
        cache_entry = {
            "data": data,
            "cached_at": datetime.now().isoformat(),
            "key": cache_key,
        }
        with open(file_path, "w") as f:
            json.dump(cache_entry, f, indent=2)
        logger.info(f"Saved to file cache: {cache_key}")
        return True
    except Exception as e:
        logger.error(f"Error saving to file cache: {e}")
        return False

@app.post("/api/demand_forecast_ratio")
async def get_demand_forecast_ratio(req: Request):
    """ 
    Get the ratio between forecasted demand and active drivers.
    This helps evaluate the supply-demand balance for each region.
    """
    try:
        # Get query parameters
        body = await req.json()
        data_type = body.get("type", "ward")  # 'ward' or 'ca'
        hours = int(body.get("hours", "24"))
        region = body.get("region")
        force_refresh = body.get("refresh", "false")
        if isinstance(force_refresh, bool):
            # It's already a boolean
            pass
        elif isinstance(force_refresh, str):
            force_refresh = force_refresh == "true"
        else:
            force_refresh = False
        
        print("request_args: ", body)

        # Create cache key
        cache_key = f"demand_driver_ratio_{data_type}_{hours}"
        print("cache_key: ", cache_key)

        # Load data from file instead of calculating
        try:
            # Check multiple possible locations for the file
            possible_paths = [
                f"demand_driver_ratio_{data_type}_{hours}.json",
                os.path.join("data", f"demand_driver_ratio_{data_type}_{hours}.json"),
                os.path.join("..", f"demand_driver_ratio_{data_type}_{hours}.json"),
                os.path.join(os.getcwd(), f"demand_driver_ratio_{data_type}_{hours}.json")
            ]
            
            file_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    file_path = path
                    break
            
            if file_path:
                logger.info(f"Found data file at: {file_path}")
                with open(file_path, "r") as f:
                    file_data = json.load(f)
                    
                hourly_ratios = file_data.get("data", {}).get("hourly_ratios", {})
                if region in hourly_ratios and hourly_ratios[region]:
                    # Return only the first record for the specified region
                    return hourly_ratios[region][0]
                else:
                    return {"error": f"No data available for region {region}"}
            else:
                # Print the current working directory to help debug
                cwd = os.getcwd()
                logger.error(f"Current working directory: {cwd}")
                logger.error(f"Could not find data file. Tried paths: {possible_paths}")
                return {"error": f"Data file not found: {possible_paths[0]}. Current directory: {cwd}"}
                
        except Exception as e:
            logger.error(f"Error loading data from file: {e}")
            logger.error(traceback.format_exc())
            return {"error": f"Error loading data: {str(e)}"}

    except Exception as e:
        logger.error(f"Error in demand forecast ratio: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}


@app.post("/api/surge_pricing")
async def get_surge_pricing(req: Request):
    """
    Calculate the final price with surge for a given trip.
    """
    try:
        # Get query parameters
        body = await req.json()
        data_type = body.get("type", "ward")  # 'ward' or 'ca'
        region = body.get("region")
        distance = float(body.get("distance", "5.0"))  # Trip distance in km
        duration = float(body.get("duration", "20.0"))  # Trip duration in minutes
        alpha = float(body.get("alpha", "0.5"))  # Surge sensitivity parameter
        surge = bool(body.get("surge"))  # Surge multiplier enabled

        # Get demand/supply data from the stored JSON file
        # Check multiple possible locations for the file
        possible_paths = [
            f"demand_driver_ratio_{data_type}_{24}.json",
            os.path.join("data", f"demand_driver_ratio_{data_type}_{24}.json"),
            os.path.join("..", f"demand_driver_ratio_{data_type}_{24}.json"),
            os.path.join(os.getcwd(), f"demand_driver_ratio_{data_type}_{24}.json")
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        try:
            if file_path:
                logger.info(f"Found data file at: {file_path}")
                with open(file_path, "r") as f:
                    data = json.load(f)

                hourly_ratios = data.get("data", {}).get("hourly_ratios", {})
                if region not in hourly_ratios or not hourly_ratios[region]:
                    return {"error": f"No data available for region {region}"}
                
                # Get the first record for this region
                region_data = hourly_ratios[region][0]
                demand = region_data.get("forecast_requests", 0)
                supply = region_data.get("active_drivers", 1)  # Default to 1 to avoid division by zero
                
                # Calculate base fare
                base_fare = MINIMUM_PRICE + (PER_KM_VALUE * distance)
                time_fare = duration * PER_MIN_CHARGE
                subtotal = base_fare + time_fare

                # Calculate surge multiplier
                demand_supply_ratio = demand / supply if supply > 0 else 1.0
                surge_multiplier = 1 + alpha * (demand_supply_ratio - 1)
                
                if surge == False:
                    surge_multiplier = 1.0
                else: 
                    surge_multiplier = max(min(2, round(surge_multiplier, 2)), 1)

                # Calculate final price with surge
                total_price = subtotal * surge_multiplier

                # Prepare response
                return {
                    "pricing": {
                        "base_fare": round(base_fare, 2),
                        "time_fare": round(time_fare, 2),
                        "subtotal": round(subtotal, 2),
                        "surge_multiplier": surge_multiplier,
                        "total_price": round(total_price, 2)
                    },
                    "trip_details": {
                        "distance_km": distance,
                        "duration_min": duration,
                        "region": region,
                        "pricing_time": region_data.get("datetime")
                    },
                    "demand_supply": {
                        "forecast_requests": demand,
                        "active_drivers": supply,
                        "available_drivers": region_data.get("available_drivers", 0),
                        "demand_supply_ratio": round(demand_supply_ratio, 2)
                    },
                    "pricing_constants": {
                        "minimum_price": MINIMUM_PRICE,
                        "per_km_value": PER_KM_VALUE,
                        "per_min_charge": PER_MIN_CHARGE,
                        "alpha": alpha
                    }
                }
            else:
                # Print the current working directory to help debug
                cwd = os.getcwd()
                logger.error(f"Current working directory: {cwd}")
                logger.error(f"Could not find data file. Tried paths: {possible_paths}")
                return {"error": f"Data file not found. Current directory: {cwd}"}
                
        except Exception as e:
            logger.error(f"Error loading data from file: {e}")
            logger.error(traceback.format_exc())
            return {"error": f"Error loading data: {str(e)}"}

    except Exception as e:
        logger.error(f"Error calculating surge pricing: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8888)