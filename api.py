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
        x,
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

        # Check cache first
        cache_key = f"forecast_{data_type}_{hours}_{region}"
        with cache_lock:
            if cache_key in forecast_cache:
                cache_entry = forecast_cache[cache_key]
                cache_time = cache_entry["timestamp"]
                # Use cache if it's less than 30 minutes old
                if (datetime.now() - cache_time).total_seconds() < 1800:
                    logger.info(f"Using cached forecast for {cache_key}")
                    return jsonify(cache_entry["data"])

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

        # Cache the result
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
@timed_cache(timeout_seconds=300)  # Cache for 5 minutes
def get_regions():
    """
    Get available regions for a given type.
    """
    try:
        data_type = request.args.get("type", "ward")
        filter_type = request.args.get("filter", "all")  # 'all', 'simple', 'b', 'w'

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

        return jsonify({"regions": regions})
    except Exception as e:
        logger.error(f"Error getting regions: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/historical", methods=["GET"])
@timed_cache(timeout_seconds=300)  # Cache for 5 minutes
def get_historical_data():
    """Get historical data for the specified parameters"""
    try:
        # Get query parameters
        data_type = request.args.get("type", "ward")  # 'ward' or 'ca'
        region = request.args.get("region")
        hours = int(request.args.get("hours", "48"))  # Default to last 48 hours

        # Limit hours to prevent large responses
        if hours > 168:  # 1 week
            hours = 168
            logger.warning(f"Historical hours capped at 168 (requested: {hours})")

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

        # Get historical data
        df = data_manager.get_historical_trends_data(data_type)

        if df.empty:
            return jsonify({"error": "No historical data available"}), 404

        # Determine region column based on data type
        region_col = "ward_num" if data_type == "ward" else "ac_num"

        # Filter by region if specified
        if region is not None and region_col in df.columns:
            df = df[df[region_col] == region]

        # Filter to recent data
        if "datetime" in df.columns:
            min_datetime = datetime.now() - timedelta(hours=hours)
            df = df[df["datetime"] >= min_datetime]

        # Format the response
        result = []
        for _, row in df.iterrows():
            record = {
                "datetime": (
                    row["datetime"].isoformat() if "datetime" in df.columns else None
                ),
                "region": row[region_col] if region_col in df.columns else None,
                "search_requests": (
                    int(row["srch_rqst"]) if "srch_rqst" in df.columns else 0
                ),
            }
            result.append(record)

        return jsonify(
            {
                "historical_data": result,
                "metadata": {
                    "data_type": data_type,
                    "hours": hours,
                    "region": region,
                    "count": len(result),
                },
            }
        )

    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/model/info", methods=["GET"])
@timed_cache(timeout_seconds=600)  # Cache for 10 minutes
def get_model_info():
    """Get information about the trained model"""
    try:
        data_type = request.args.get("type", "ward")  # 'ward' or 'ca'

        # Check if model is loaded
        if forecaster.model is None:
            forecaster.load_model(data_type)

        if forecaster.model is None:
            return jsonify({"error": f"No model available for {data_type}"}), 404

        # Get feature importance if available
        feature_importance = []
        if forecaster.model is not None and forecaster.features:
            importances = forecaster.model.feature_importances_
            for i, feature in enumerate(forecaster.features):
                feature_importance.append(
                    {"feature": feature, "importance": float(importances[i])}
                )

            # Sort by importance
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        # Get model metadata
        model_path = os.path.join("data/models", f"rf_model_{data_type}.pkl")
        model_stats = {}
        if os.path.exists(model_path):
            model_stats["file_size"] = os.path.getsize(model_path)
            model_stats["last_modified"] = datetime.fromtimestamp(
                os.path.getmtime(model_path)
            ).isoformat()

        return jsonify(
            {
                "model_type": "RandomForestRegressor",
                "data_type": data_type,
                "features": forecaster.features,
                "feature_importance": feature_importance,
                "model_stats": model_stats,
                "n_estimators": (
                    forecaster.model.n_estimators if forecaster.model else None
                ),
                "max_depth": forecaster.model.max_depth if forecaster.model else None,
            }
        )

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/retrain", methods=["POST"])
def retrain_model():
    """Retrain the model with the latest data"""
    try:
        data_type = request.args.get("type", "ward")  # 'ward' or 'ca'

        # Fetch latest data
        data_manager.fetch_all_endpoints()

        # Retrain model
        forecaster.train_model(data_type, force=True)

        if forecaster.model is None:
            return jsonify({"error": "Failed to train model"}), 500

        # Clear cache after retraining
        with cache_lock:
            forecast_cache.clear()

        return jsonify(
            {
                "status": "success",
                "message": f"Model for {data_type} retrained successfully",
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """Clear the API cache"""
    try:
        with cache_lock:
            forecast_cache.clear()

        return jsonify(
            {
                "status": "success",
                "message": "Cache cleared successfully",
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        logger.error(traceback.format_exc())
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

        # Set reasonable limits
        if hours > 72:
            hours = 72  # Cap at 72 hours to prevent long processing times
            logger.warning(f"Forecast hours capped at 72 (requested: {hours})")

        if max_workers > 8:
            max_workers = 8  # Cap at 8 workers to prevent overloading the server
            logger.warning(f"Worker count capped at 8 (requested: {max_workers})")

        # Check cache first
        cache_key = f"forecast_all_{data_type}_{hours}_{filter_type}"
        with cache_lock:
            if cache_key in forecast_cache:
                cache_entry = forecast_cache[cache_key]
                cache_time = cache_entry["timestamp"]
                # Use cache if it's less than 30 minutes old
                if (datetime.now() - cache_time).total_seconds() < 1800:
                    logger.info(f"Using cached forecast for {cache_key}")
                    return jsonify(cache_entry["data"])

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

        # Cache the result
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


if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs("logs", exist_ok=True)

    # Run the Flask app
    app.run(host="0.0.0.0", port=8888, debug=True, threaded=True)
