import os
import json
import logging
import traceback
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import threading
import functools
from werkzeug.serving import run_simple
import re
import concurrent.futures
import time

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

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/api/forecast", methods=["GET"])
def get_forecast():
    """Get forecast for the specified parameters"""
    try:
        # Get query parameters
        data_type = request.args.get("type", "ward")  # 'ward' or 'ca'
        hours = int(request.args.get("hours", "24"))
        region = request.args.get("region")
        force_refresh = request.args.get("refresh", "false").lower() == "true"

        # No longer convert region to integer - keep it as a string
        # This allows for region identifiers like "b_1" or "w_0"
        if region is not None:
            # Validate that the region exists
            df = data_manager.get_historical_trends_data(data_type)
            region_col = "ward_num" if data_type == "ward" else "ac_num"

            if region_col in df.columns:
                valid_regions = df[region_col].unique().tolist()
                if region not in valid_regions:
                    logger.warning(
                        f"Region {region} not found in data. Valid regions: {valid_regions[:5]}..."
                    )
                    return (
                        jsonify(
                            {
                                "error": f"Region {region} not found. Valid regions include: {valid_regions[:5]}..."
                            }
                        ),
                        400,
                    )

        # Create cache key
        cache_key = f"forecast_{data_type}_{hours}_{region}"

        # Check memory cache first if not forcing refresh
        if not force_refresh:
            with cache_lock:
                if cache_key in forecast_cache:
                    cache_entry = forecast_cache[cache_key]
                    cache_time = cache_entry["timestamp"]
                    # Use cache if it's less than 30 minutes old
                    if (datetime.now() - cache_time).total_seconds() < 1800:
                        logger.info(f"Using memory cache for {cache_key}")
                        return jsonify(cache_entry["data"])

            # Check file cache for this specific region
            file_cache_data = load_from_file_cache(cache_key)
            if file_cache_data:
                # Also update memory cache
                with cache_lock:
                    forecast_cache[cache_key] = {
                        "data": file_cache_data,
                        "timestamp": datetime.now(),
                    }
                logger.info(f"Using file cache for specific region: {cache_key}")
                return jsonify(file_cache_data)

            # Also check if this data might be available from the "forecast/all" endpoint's cache
            # Look for cached data from forecast/all that includes this region
            all_forecast_cache_key = f"forecast_all_{data_type}_{hours}_all"
            all_forecast_cache_data = load_from_file_cache(all_forecast_cache_key)

            if all_forecast_cache_data and "forecasts" in all_forecast_cache_data:
                # Check if this region is included in the all-regions forecast
                if region in all_forecast_cache_data["forecasts"]:
                    logger.info(
                        f"Using region {region} data from all-regions forecast cache"
                    )

                    # Extract this region's data from the all-regions forecast
                    region_data = all_forecast_cache_data["forecasts"][region]

                    # Format response similar to single region forecast
                    response_data = {
                        "forecast": [
                            {
                                "datetime": item["datetime"],
                                "region": region,
                                "forecast_requests": item["forecast_requests"],
                            }
                            for item in region_data
                        ],
                        "metadata": {
                            "data_type": data_type,
                            "hours": hours,
                            "region": region,
                            "generated_at": all_forecast_cache_data["metadata"][
                                "generated_at"
                            ],
                            "source": "all_regions_cache",
                        },
                    }

                    # Also cache this separately for faster access next time
                    save_to_file_cache(cache_key, response_data)

                    # Update memory cache
                    with cache_lock:
                        forecast_cache[cache_key] = {
                            "data": response_data,
                            "timestamp": datetime.now(),
                        }

                    return jsonify(response_data)

        # Set a reasonable limit on forecast hours
        if hours > 72:
            hours = 72  # Cap at 72 hours to prevent long processing times
            logger.warning(f"Forecast hours capped at 72 (requested: {hours})")

        # Generate forecast with a timeout
        logger.info(
            f"Generating forecast for {data_type}, region {region}, hours {hours}"
        )

        # For optimization, reduce the number of estimators for quicker forecasts
        original_n_estimators = None
        if forecaster.model is not None and hasattr(forecaster.model, "n_estimators"):
            original_n_estimators = forecaster.model.n_estimators
            if original_n_estimators > 50:
                forecaster.model.n_estimators = 50
                logger.info(
                    f"Temporarily reduced n_estimators from {original_n_estimators} to 50 for faster forecasting"
                )

        try:
            forecast_df = forecaster.forecast_next_hours(hours, data_type, region)
        finally:
            # Restore original n_estimators
            if original_n_estimators is not None:
                forecaster.model.n_estimators = original_n_estimators

        if forecast_df is None or forecast_df.empty:
            return jsonify({"error": "Failed to generate forecast"}), 500

        # Convert DataFrame to JSON
        region_col = "ward_num" if data_type == "ward" else "ac_num"

        # Format the response
        result = []
        for _, row in forecast_df.iterrows():
            result.append(
                {
                    "datetime": row["datetime"].isoformat(),
                    "region": row[region_col],
                    "forecast_requests": int(row["forecast_requests"]),
                }
            )

        response_data = {
            "forecast": result,
            "metadata": {
                "data_type": data_type,
                "hours": hours,
                "region": region,
                "generated_at": datetime.now().isoformat(),
            },
        }

        # Save to file cache
        save_to_file_cache(cache_key, response_data)

        # Cache in memory
        with cache_lock:
            forecast_cache[cache_key] = {
                "data": response_data,
                "timestamp": datetime.now(),
            }

            # Clean old cache entries
            current_time = datetime.now()
            keys_to_remove = []
            for key, entry in forecast_cache.items():
                if (current_time - entry["timestamp"]).total_seconds() > 3600:  # 1 hour
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del forecast_cache[key]

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/regions", methods=["GET"])
def get_regions():
    """
    Get available regions for a given type.
    """
    try:
        data_type = request.args.get("type", "ward")
        filter_type = request.args.get("filter", "all")  # 'all', 'simple', 'b', 'w'
        force_refresh = request.args.get("refresh", "false").lower() == "true"

        # Create cache key
        cache_key = f"regions_{data_type}_{filter_type}"

        # Check file cache if not forcing refresh
        if not force_refresh:
            file_cache_data = load_from_file_cache(cache_key)
            if file_cache_data:
                return jsonify(file_cache_data)

        # Get the regions from the historical data
        regions = []

        # Get the historical data
        df = data_manager.get_historical_trends_data(data_type)

        if not df.empty:
            # Get unique regions
            region_col = "region"  # We're now using a consistent region column name
            regions = df[region_col].unique().tolist()

            # Apply filtering if requested
            if filter_type != "all":
                if filter_type == "simple":
                    # Filter for simple numeric regions (1, 2, 3, etc.)
                    regions = [r for r in regions if str(r).isdigit()]
                elif filter_type == "b":
                    # Filter for b_ prefixed regions
                    regions = [r for r in regions if str(r).startswith("b_")]
                elif filter_type == "w":
                    # Filter for w_ prefixed regions
                    regions = [r for r in regions if str(r).startswith("w_")]

            # Convert all regions to strings for consistency
            regions = [str(r) for r in regions]

            # Sort regions naturally (1, 2, 10 instead of 1, 10, 2)
            regions.sort(
                key=lambda x: [
                    int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)
                ]
            )

            logger.info(
                f"Found {len(regions)} regions for type {data_type} with filter {filter_type}"
            )
        else:
            logger.warning(f"No historical data found for type {data_type}")

        response_data = {"regions": regions}

        # Save to file cache
        save_to_file_cache(cache_key, response_data)

        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error getting regions: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/forecast/all", methods=["GET"])
def get_all_forecasts():
    """Get forecast for all regions of the specified type"""
    try:
        # Get query parameters
        data_type = request.args.get("type", "ward")  # 'ward' or 'ca'
        hours = int(request.args.get("hours", "24"))
        filter_type = request.args.get(
            "filter", "all"
        )  # Optional filter for region types
        max_workers = int(
            request.args.get("workers", "4")
        )  # Number of parallel workers
        batch_size = int(
            request.args.get("batch_size", "0")
        )  # Optional batch size for processing
        force_refresh = request.args.get("refresh", "false").lower() == "true"

        # Set reasonable limits
        if hours > 72:
            hours = 72  # Cap at 72 hours to prevent long processing times
            logger.warning(f"Forecast hours capped at 72 (requested: {hours})")

        if max_workers > 8:
            max_workers = 8  # Cap at 8 workers to prevent overloading the server
            logger.warning(f"Worker count capped at 8 (requested: {max_workers})")

        # Create cache key
        cache_key = f"forecast_all_{data_type}_{hours}_{filter_type}"
        if batch_size > 0:
            cache_key += f"_batch{batch_size}"

        # Check memory cache first if not forcing refresh
        if not force_refresh:
            with cache_lock:
                if cache_key in forecast_cache:
                    cache_entry = forecast_cache[cache_key]
                    cache_time = cache_entry["timestamp"]
                    # Use cache if it's less than 30 minutes old
                    if (datetime.now() - cache_time).total_seconds() < 1800:
                        logger.info(f"Using memory cache for {cache_key}")
                        return jsonify(cache_entry["data"])

            # Check file cache next
            file_cache_data = load_from_file_cache(cache_key)
            if file_cache_data:
                # Also update memory cache
                with cache_lock:
                    forecast_cache[cache_key] = {
                        "data": file_cache_data,
                        "timestamp": datetime.now(),
                    }
                return jsonify(file_cache_data)

        # Get regions directly from data manager instead of calling the endpoint
        df = data_manager.get_historical_trends_data(data_type)
        if df.empty:
            return (
                jsonify({"error": f"No historical data found for type {data_type}"}),
                404,
            )

        # Get unique regions
        region_col = "ward_num" if data_type == "ward" else "ac_num"
        all_regions = df[region_col].unique().tolist()

        # Convert all regions to strings for consistency
        all_regions = [str(r) for r in all_regions]

        # Apply filter if specified
        if filter_type != "all":
            if filter_type == "simple":
                all_regions = [r for r in all_regions if str(r).isdigit()]
            elif filter_type == "b":
                all_regions = [r for r in all_regions if str(r).startswith("b_")]
            elif filter_type == "w":
                all_regions = [r for r in all_regions if str(r).startswith("w_")]

        if not all_regions:
            return (
                jsonify({"error": f"No regions found with filter: {filter_type}"}),
                404,
            )

        # Sort regions naturally (1, 2, 10 instead of 1, 10, 2)
        all_regions.sort(
            key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)]
        )

        # Apply batch size if specified
        if batch_size > 0 and batch_size < len(all_regions):
            logger.info(
                f"Using batch size {batch_size} instead of processing all {len(all_regions)} regions"
            )
            # Take the first batch_size regions
            all_regions = all_regions[:batch_size]

        logger.info(
            f"Generating forecasts for {len(all_regions)} regions, type {data_type}, hours {hours} using {max_workers} workers"
        )

        # For optimization, reduce the number of estimators for quicker forecasts
        original_n_estimators = None
        if forecaster.model is not None and hasattr(forecaster.model, "n_estimators"):
            original_n_estimators = forecaster.model.n_estimators
            if original_n_estimators > 50:
                forecaster.model.n_estimators = 50
                logger.info(
                    f"Temporarily reduced n_estimators from {original_n_estimators} to 50 for faster forecasting"
                )

        # Function to generate forecast for a single region
        def generate_region_forecast(region):
            try:
                thread_id = threading.get_ident()
                logger.info(
                    f"Worker {thread_id} generating forecast for region {region}"
                )
                start_time = time.time()

                # Check if individual region forecast is cached
                region_cache_key = f"forecast_{data_type}_{hours}_{region}"
                region_cache_data = load_from_file_cache(region_cache_key)

                if region_cache_data and not force_refresh:
                    logger.info(
                        f"Worker {thread_id}: Using cached forecast for region {region}"
                    )
                    region_results = region_cache_data.get("forecast", [])
                    return region, region_results

                # Otherwise generate the forecast
                region_forecast_df = forecaster.forecast_next_hours(
                    hours, data_type, region
                )

                elapsed = time.time() - start_time

                if region_forecast_df is None or region_forecast_df.empty:
                    logger.warning(
                        f"Worker {thread_id}: Failed to generate forecast for region {region}"
                    )
                    return region, None

                # Convert DataFrame to list of dictionaries
                region_col = "ward_num" if data_type == "ward" else "ac_num"
                region_results = []

                for _, row in region_forecast_df.iterrows():
                    region_results.append(
                        {
                            "datetime": row["datetime"].isoformat(),
                            "forecast_requests": int(row["forecast_requests"]),
                        }
                    )

                # Cache individual region forecast
                region_response_data = {
                    "forecast": region_results,
                    "metadata": {
                        "data_type": data_type,
                        "hours": hours,
                        "region": region,
                        "generated_at": datetime.now().isoformat(),
                    },
                }
                save_to_file_cache(region_cache_key, region_response_data)

                logger.info(
                    f"Worker {thread_id}: Completed forecast for region {region} with {len(region_results)} points in {elapsed:.2f}s"
                )
                return region, region_results
            except Exception as e:
                logger.error(
                    f"Worker error generating forecast for region {region}: {e}"
                )
                logger.error(traceback.format_exc())
                return region, None

        try:
            # Use ThreadPoolExecutor for parallel processing
            import concurrent.futures

            results_by_region = {}
            completed_count = 0
            total_regions = len(all_regions)

            # Create a progress tracking function
            def log_progress():
                if completed_count > 0:
                    percent_complete = (completed_count / total_regions) * 100
                    logger.info(
                        f"Progress: {completed_count}/{total_regions} regions completed ({percent_complete:.1f}%)"
                    )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Submit all forecast tasks
                future_to_region = {
                    executor.submit(generate_region_forecast, region): region
                    for region in all_regions
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_region):
                    region = future_to_region[future]
                    try:
                        region_key, region_data = future.result()
                        if region_data:
                            results_by_region[region_key] = region_data

                        # Update progress
                        completed_count += 1
                        if (
                            completed_count % max(1, total_regions // 10) == 0
                            or completed_count == total_regions
                        ):
                            log_progress()

                    except Exception as e:
                        logger.error(
                            f"Worker for region {region} raised exception: {e}"
                        )
                        logger.error(traceback.format_exc())
                        completed_count += 1
        finally:
            # Restore original n_estimators
            if original_n_estimators is not None:
                forecaster.model.n_estimators = original_n_estimators

        if not results_by_region:
            return (
                jsonify({"error": "Failed to generate forecasts for any region"}),
                500,
            )

        response_data = {
            "forecasts": results_by_region,
            "metadata": {
                "data_type": data_type,
                "hours": hours,
                "filter": filter_type,
                "region_count": len(results_by_region),
                "total_regions_requested": len(all_regions),
                "workers": max_workers,
                "generated_at": datetime.now().isoformat(),
                "success_rate": f"{len(results_by_region)}/{len(all_regions)} ({(len(results_by_region)/len(all_regions))*100:.1f}%)",
            },
        }

        # Save to file cache
        save_to_file_cache(cache_key, response_data)

        # Cache in memory
        with cache_lock:
            forecast_cache[cache_key] = {
                "data": response_data,
                "timestamp": datetime.now(),
            }

            # Clean old cache entries
            current_time = datetime.now()
            keys_to_remove = []
            for key, entry in forecast_cache.items():
                if (current_time - entry["timestamp"]).total_seconds() > 3600:  # 1 hour
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del forecast_cache[key]

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error generating all forecasts: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/demand_forecast_ratio", methods=["GET"])
def get_demand_forecast_ratio():
    """
    Get the ratio between forecasted demand and active drivers.
    This helps evaluate the supply-demand balance for each region.
    """
    try:
        # Get query parameters
        data_type = request.args.get("type", "ward")  # 'ward' or 'ca'
        hours = int(request.args.get("hours", "24"))
        region = request.args.get("region")
        force_refresh = request.args.get("refresh", "false").lower() == "true"

        # Create cache key
        cache_key = f"demand_driver_ratio_{data_type}_{hours}_{region}"

        # Check file cache if not forcing refresh
        if not force_refresh:
            file_cache_data = load_from_file_cache(cache_key)
            if file_cache_data:
                return jsonify(file_cache_data)

        # Load the active driver data from the JSON file
        driver_data = {}
        try:
            file_path = os.path.join("data", f"driver_eda_{data_type}s_new_key.json")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    driver_info = json.load(f)

                # Convert to dictionary with ward_num/ac_num as keys
                region_col = "ward_num" if data_type == "ward" else "ac_num"

                # Aggregate driver data by region
                for item in driver_info:
                    if region_col in item and "active_drvr" in item:
                        region_id = str(item[region_col])

                        if region_id not in driver_data:
                            driver_data[region_id] = {
                                "active_drvr": 0,
                                "drvr_notonride": 0,
                                "drvr_onride": 0,
                                "region": region_id,
                            }

                        # Sum up driver counts
                        driver_data[region_id]["active_drvr"] += int(
                            item.get("active_drvr", 0)
                        )
                        driver_data[region_id]["drvr_notonride"] += int(
                            item.get("drvr_notonride", 0)
                        )
                        driver_data[region_id]["drvr_onride"] += int(
                            item.get("drvr_onride", 0)
                        )
            else:
                logger.warning(f"Driver data file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading driver data: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Error loading driver data: {str(e)}"}), 500

        if not driver_data:
            return jsonify({"error": "No driver data available"}), 404

        # Generate forecast for the same regions
        forecast_df = forecaster.forecast_next_hours(hours, data_type, region)

        if forecast_df is None or forecast_df.empty:
            return jsonify({"error": "Failed to generate forecast"}), 500

        # Calculate ratios hour by hour
        region_col = "ward_num" if data_type == "ward" else "ac_num"

        # Prepare the result structure
        hourly_ratios = {}

        # Process each hour of the forecast
        for _, row in forecast_df.iterrows():
            region_id = str(row[region_col])
            hour = row["datetime"]
            forecast_requests = row["forecast_requests"]

            if region_id in driver_data:
                active_drivers = driver_data[region_id]["active_drvr"]
                available_drivers = driver_data[region_id]["drvr_notonride"]

                # Calculate ratios
                demand_supply_ratio = 0
                if active_drivers > 0:
                    demand_supply_ratio = forecast_requests / active_drivers

                available_ratio = 0
                if available_drivers > 0:
                    available_ratio = forecast_requests / available_drivers

                # Initialize region in hourly_ratios if not exists
                if region_id not in hourly_ratios:
                    hourly_ratios[region_id] = []

                # Add hourly data
                hourly_ratios[region_id].append(
                    {
                        "datetime": hour.isoformat(),
                        "forecast_requests": int(forecast_requests),
                        "active_drivers": active_drivers,
                        "available_drivers": available_drivers,
                        "demand_supply_ratio": round(demand_supply_ratio, 2),
                        "available_ratio": round(available_ratio, 2),
                    }
                )

        # Calculate overall statistics
        all_hours_data = []
        for region_data in hourly_ratios.values():
            all_hours_data.extend(region_data)

        total_forecast = sum(item["forecast_requests"] for item in all_hours_data)
        total_drivers = (
            sum(item["active_drivers"] for item in all_hours_data) / len(all_hours_data)
            if all_hours_data
            else 0
        )
        total_available = (
            sum(item["available_drivers"] for item in all_hours_data)
            / len(all_hours_data)
            if all_hours_data
            else 0
        )

        overall_ratio = 0
        if total_drivers > 0:
            overall_ratio = total_forecast / (total_drivers * len(hourly_ratios))

        available_ratio = 0
        if total_available > 0:
            available_ratio = total_forecast / (total_available * len(hourly_ratios))

        # Find peak demand hours
        peak_hours = []
        if all_hours_data:
            # Group by hour
            hour_groups = {}
            for item in all_hours_data:
                hour = datetime.fromisoformat(item["datetime"]).hour
                if hour not in hour_groups:
                    hour_groups[hour] = []
                hour_groups[hour].append(item)

            # Calculate average demand for each hour
            hour_demand = {}
            for hour, items in hour_groups.items():
                hour_demand[hour] = sum(
                    item["forecast_requests"] for item in items
                ) / len(items)

            # Get top 3 peak hours
            peak_hours = sorted(hour_demand.items(), key=lambda x: x[1], reverse=True)[
                :3
            ]
            peak_hours = [
                {"hour": hour, "avg_demand": round(demand, 2)}
                for hour, demand in peak_hours
            ]

        response_data = {
            "hourly_ratios": hourly_ratios,
            "summary": {
                "total_forecast_requests": total_forecast,
                "avg_active_drivers": round(total_drivers, 2),
                "avg_available_drivers": round(total_available, 2),
                "overall_demand_supply_ratio": round(overall_ratio, 2),
                "overall_available_ratio": round(available_ratio, 2),
                "regions_count": len(hourly_ratios),
                "hours_count": hours,
                "peak_hours": peak_hours,
            },
            "metadata": {
                "data_type": data_type,
                "hours": hours,
                "region": region,
                "generated_at": datetime.now().isoformat(),
                "driver_data_source": file_path,
            },
        }

        # Save to file cache
        save_to_file_cache(cache_key, response_data)

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error calculating demand-driver ratio: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs("logs", exist_ok=True)

    # Run the Flask app
    app.run(host="0.0.0.0", port=8888, debug=True, threaded=True)
