import uvicorn
from datetime import datetime
import os
import threading
import re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from data_manager import RideDataManager
from forecaster import RideRequestForecast
from cache_manager import (
    load_from_file_cache,
    save_to_file_cache,
)
from custom_logger import CustomLogger
import traceback

logger = CustomLogger("forecast_api", "logs/api_log.txt")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

data_manager = RideDataManager()
forecaster = RideRequestForecast(data_manager)

forecast_cache = {}
cache_lock = threading.Lock()

CACHE_DIR = "cache/forecasts"
os.makedirs(CACHE_DIR, exist_ok=True)

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

MINIMUM_PRICE = 30
PER_KM_VALUE = 15
PER_MIN_CHARGE = 1.5

endpoints = [
    "trends_live_ward_new_key.json",
    "trends_live_ca_new_key.json",
    "driver_eda_wards_new_key.json",
    "driver_eda_ca_new_key.json",
    "funnel_live_ward_new_key.json",
    "funnel_live_ca_new_key.json",
    "funnel_cumulative_ward_new_key.json",
    "funnel_cumulative_ca_new_key.json",
    "cumulative_stats_new_key.json",
]
base_url = "https://d11gklsvr97l1g.cloudfront.net/open/json-data"


def init():
    """Initialize the application by loading data and models."""
    try:
        logger.info("Loading historical data and models...")
        data_manager.load_historical_data()

        logger.info("Fetching current data from endpoints...")
        fetched_data = data_manager.fetch_all_endpoints(endpoints, base_url)
        if fetched_data:
            logger.info("Successfully fetched current data")
        else:
            logger.warning(
                "Failed to fetch current data, will use cached data if available"
            )

        if not forecaster.load_model("ward"):
            logger.info("No pre-trained model found, training a new one...")
            forecaster.train_model("ward")
        logger.info("Historical data and models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading data or models: {e}")
        logger.error(traceback.format_exc())


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    logger.info(f"Health check endpoint accessed at {datetime.now()}")
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/forecast")
async def get_forecast(
    data_type: str = "ward", hours: int = 24, region: str = None, refresh: bool = False
):
    """Get forecast for the specified parameters"""
    try:
        logger.debug(
            f"Forecast request parameters: data_type={data_type}, hours={hours}, region={region}"
        )
        if region is not None:
            df = data_manager.get_historical_trends_data(data_type)
            region_col = "ward_num" if data_type == "ward" else "ac_num"
            if region_col in df.columns:
                valid_regions = df[region_col].unique().tolist()
                if region not in valid_regions:
                    logger.warning(
                        f"Region {region} not found in data. Valid regions: {valid_regions[:5]}..."
                    )
                    return {
                        "error": f"Region {region} not found. Valid regions include: {valid_regions[:5]}..."
                    }, 400

        cache_key = f"forecast_{data_type}_{hours}_{region}"

        if not refresh:
            with cache_lock:
                if cache_key in forecast_cache:
                    cache_entry = forecast_cache[cache_key]
                    cache_time = cache_entry["timestamp"]
                    if (datetime.now() - cache_time).total_seconds() < 1800:
                        logger.info(f"Using memory cache for {cache_key}")
                        return cache_entry["data"]

            file_cache_data = load_from_file_cache(cache_key)
            if file_cache_data:
                with cache_lock:
                    forecast_cache[cache_key] = {
                        "data": file_cache_data,
                        "timestamp": datetime.now(),
                    }
                logger.info(f"Using file cache for specific region: {cache_key}")
                return file_cache_data

        if hours > 72:
            hours = 72
            logger.warning(f"Forecast hours capped at 72 (requested: {hours}")

        logger.info(
            f"Generating forecast for {data_type}, region {region}, hours {hours}"
        )

        forecast_df = forecaster.forecast_next_hours(hours, data_type, region)

        if forecast_df is None or forecast_df.empty:
            return {"error": "Failed to generate forecast"}, 500

        region_col = "ward_num" if data_type == "ward" else "ac_num"

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

        save_to_file_cache(cache_key, response_data)

        with cache_lock:
            forecast_cache[cache_key] = {
                "data": response_data,
                "timestamp": datetime.now(),
            }

        return response_data

    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        return {"error": str(e)}, 500


@app.get("/api/regions")
async def get_regions(
    data_type: str = "ward", filter_type: str = "all", refresh: bool = False
):
    """Get available regions for a given type."""
    try:
        logger.debug(
            f"Regions request parameters: data_type={data_type}, filter_type={filter_type}"
        )
        cache_key = f"regions_{data_type}_{filter_type}"

        if not refresh:
            file_cache_data = load_from_file_cache(cache_key)
            if file_cache_data:
                return file_cache_data

        regions = []

        df = data_manager.get_historical_trends_data(data_type)

        if not df.empty:
            region_col = "region"
            regions = df[region_col].unique().tolist()

            if filter_type != "all":
                if filter_type == "simple":
                    regions = [r for r in regions if str(r).isdigit()]
                elif filter_type == "b":
                    regions = [r for r in regions if str(r).startswith("b_")]
                elif filter_type == "w":
                    regions = [r for r in regions if str(r).startswith("w_")]

            regions = [str(r) for r in regions]

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

        save_to_file_cache(cache_key, response_data)

        return response_data
    except Exception as e:
        logger.error(f"Error getting regions: {e}")
        return {"error": str(e)}, 500


@app.get("/api/forecast/all")
async def get_all_forecasts(
    data_type: str = "ward",
    hours: int = 24,
    filter_type: str = "all",
    max_workers: int = 4,
    batch_size: int = 0,
    refresh: bool = False,
):
    """Get forecast for all regions of the specified type"""
    try:
        logger.debug(
            f"All forecasts request parameters: data_type={data_type}, hours={hours}, filter_type={filter_type}"
        )
        cache_key = f"all_forecasts_{data_type}_{hours}_{filter_type}"

        if not refresh:
            file_cache_data = load_from_file_cache(cache_key)
            if file_cache_data:
                return file_cache_data

        df = data_manager.get_historical_trends_data(data_type)

        if df.empty:
            logger.warning(f"No historical data found for type {data_type}")
            return {"error": "No historical data available"}, 404

        region_col = "ward_num" if data_type == "ward" else "ac_num"
        regions = df[region_col].unique().tolist()

        forecasts = []

        for region in regions:
            forecast_df = forecaster.forecast_next_hours(hours, data_type, region)

            if forecast_df is None or forecast_df.empty:
                logger.warning(f"Failed to generate forecast for region {region}")
                continue

            for _, row in forecast_df.iterrows():
                forecasts.append(
                    {
                        "datetime": row["datetime"].isoformat(),
                        "region": row[region_col],
                        "forecast_requests": int(row["forecast_requests"]),
                    }
                )

        response_data = {
            "forecasts": forecasts,
            "metadata": {
                "data_type": data_type,
                "hours": hours,
                "filter_type": filter_type,
                "generated_at": datetime.now().isoformat(),
            },
        }

        save_to_file_cache(cache_key, response_data)

        return response_data

    except Exception as e:
        logger.error(f"Error generating all forecasts: {e}")
        return {"error": str(e)}, 500


@app.post("/api/demand_forecast_ratio")
async def get_demand_forecast_ratio(req: Request):
    """
    Get the ratio between forecasted demand and active drivers.
    This helps evaluate the supply-demand balance for each region.
    """
    try:
        body = await req.json()
        data_type = body.get("data_type", "ward")
        hours = body.get("hours", 24)
        region = body.get("region")

        logger.debug(
            f"Demand forecast ratio request parameters: data_type={data_type}, hours={hours}, region={region}"
        )

        if not region:
            return {"error": "Region is required"}, 400

        forecast_df = forecaster.forecast_next_hours(hours, data_type, region)

        if forecast_df is None or forecast_df.empty:
            return {"error": "Failed to generate forecast"}, 500

        active_drivers = body.get("active_drivers", 0)

        if active_drivers <= 0:
            return {"error": "Active drivers must be greater than zero"}, 400

        forecasted_demand = forecast_df["forecast_requests"].sum()
        ratio = forecasted_demand / active_drivers

        logger.info(f"Calculated demand-driver ratio: {ratio} for region {region}")

        return {
            "region": region,
            "forecasted_demand": forecasted_demand,
            "active_drivers": active_drivers,
            "demand_driver_ratio": ratio,
        }

    except Exception as e:
        logger.error(f"Error calculating demand forecast ratio: {e}")
        return {"error": str(e)}, 500


@app.post("/api/surge_pricing")
async def get_surge_pricing(req: Request):
    """
    Calculate the final price with surge for a given trip.
    """
    try:
        body = await req.json()
        base_price = body.get("base_price", MINIMUM_PRICE)
        distance_km = body.get("distance_km", 0)
        duration_min = body.get("duration_min", 0)
        demand_driver_ratio = body.get("demand_driver_ratio", 1)

        logger.debug(
            f"Surge pricing request parameters: base_price={base_price}, distance_km={distance_km}, duration_min={duration_min}, demand_driver_ratio={demand_driver_ratio}"
        )

        if base_price < MINIMUM_PRICE:
            base_price = MINIMUM_PRICE

        distance_charge = distance_km * PER_KM_VALUE
        time_charge = duration_min * PER_MIN_CHARGE

        surge_multiplier = max(1, demand_driver_ratio)

        final_price = (base_price + distance_charge + time_charge) * surge_multiplier

        logger.info(f"Calculated surge pricing: final_price={final_price}")

        return {
            "base_price": base_price,
            "distance_charge": distance_charge,
            "time_charge": time_charge,
            "surge_multiplier": surge_multiplier,
            "final_price": final_price,
        }

    except Exception as e:
        logger.error(f"Error calculating surge pricing: {e}")
        return {"error": str(e)}, 500


@app.get("/api/nearby_high_demand")
async def get_nearby_high_demand(
    ward: str, max_distance: int = 5, hours: int = 3, refresh: bool = False
):
    """
    Get high-demand locations near a rider's current ward.
    Returns a list of nearby wards sorted by a score that considers both demand and distance.
    """
    try:
        logger.debug(
            f"Nearby high demand request parameters: ward={ward}, max_distance={max_distance}, hours={hours}"
        )

        
        if not re.match(r"^[a-zA-Z0-9_]+$", ward):
            logger.error(f"Invalid ward input: {ward}")
            return {
                "error": "Invalid ward input. Ward must be alphanumeric with optional underscores."
            }, 400

        if max_distance <= 0:
            logger.error(f"Invalid max_distance input: {max_distance}")
            return {
                "error": "Invalid max_distance. It must be a positive integer."
            }, 400

        cache_key = f"high_demand_{ward}_{max_distance}_{hours}"
        if not refresh:
            file_cache_data = load_from_file_cache(cache_key)
            if file_cache_data:
                return file_cache_data

        forecast_df = forecaster.forecast_next_hours(hours, "ward", None)
        if forecast_df is None or forecast_df.empty:
            logger.error("Forecast data is empty or unavailable.")
            return {"error": "Failed to generate forecast"}, 500

        
        if "ward_num" not in forecast_df.columns:
            logger.error("Missing 'ward_num' column in forecast data.")
            return {
                "error": "Forecast data is missing required 'ward_num' information."
            }, 500

        nearby_wards = data_manager.get_nearby_wards(ward, max_distance)
        if not nearby_wards:
            logger.warning(
                f"No nearby wards found for ward {ward} within {max_distance} km."
            )
            return {"error": "No nearby wards found"}, 404

        high_demand_wards = []
        for nearby_ward in nearby_wards:
            ward_forecast = forecast_df[forecast_df["ward_num"] == nearby_ward]
            if ward_forecast.empty:
                continue

            total_demand = ward_forecast["forecast_requests"].sum()
            distance = data_manager.get_distance_between_wards(ward, nearby_ward)
            score = total_demand / (distance + 1)

            high_demand_wards.append(
                {
                    "ward": nearby_ward,
                    "total_demand": total_demand,
                    "distance": distance,
                    "score": score,
                }
            )

        high_demand_wards.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Found {len(high_demand_wards)} high demand wards near {ward}")

        response_data = {"high_demand_wards": high_demand_wards}
        save_to_file_cache(cache_key, response_data)
        return response_data

    except Exception as e:
        logger.error(f"Error finding nearby high-demand locations: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}, 500


if __name__ == "__main__":

    os.makedirs("logs", exist_ok=True)
    init()

    logger.info("Starting the API server...")
    uvicorn.run(app, host="127.0.0.1", port=8888)
