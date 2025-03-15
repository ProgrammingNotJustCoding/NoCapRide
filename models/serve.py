import os
import time
import json
import logging
import pickle
import threading
import traceback
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from model import predict_and_recommend

# Ensure the logs directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/serve_log.txt", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("serve")

# Global variables
MODEL_FILE = "models/demand_model.pkl"
FEATURES_FILE = "models/model_features.pkl"
MODEL_INFO_FILE = "models/model_info.json"
DATA_DIR = "data"

# Global model objects
model = None
features = None
model_info = None
last_model_load_time = None
model_lock = threading.Lock()

# Create Flask app
app = Flask(__name__)


def load_model():
    """Load the model from disk"""
    global model, features, model_info, last_model_load_time

    try:
        with model_lock:
            # Check if model directory exists
            if not os.path.exists("models"):
                logger.warning("Models directory does not exist, creating it")
                os.makedirs("models")
                return False

            if (
                os.path.exists(MODEL_FILE)
                and os.path.exists(FEATURES_FILE)
                and os.path.exists(MODEL_INFO_FILE)
            ):
                # Load model
                with open(MODEL_FILE, "rb") as f:
                    model = pickle.load(f)

                # Load features
                with open(FEATURES_FILE, "rb") as f:
                    features = pickle.load(f)

                # Load model info
                with open(MODEL_INFO_FILE, "r") as f:
                    model_info = json.load(f)

                last_model_load_time = datetime.now()

                logger.info(
                    f"Model loaded successfully. Trained at: {model_info.get('trained_at')}"
                )
                return True
            else:
                logger.warning("Model files not found, waiting for model to be trained")
                return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        return False


def check_and_reload_model():
    """Check if the model has been updated and reload if necessary"""
    global last_model_load_time

    try:
        if os.path.exists(MODEL_INFO_FILE):
            model_file_modified = datetime.fromtimestamp(
                os.path.getmtime(MODEL_INFO_FILE)
            )

            if (
                last_model_load_time is None
                or model_file_modified > last_model_load_time
            ):
                logger.info("Model file has been updated, reloading model")
                load_model()
                return True
    except Exception as e:
        logger.error(f"Error checking model update: {e}")
        logger.error(traceback.format_exc())

    return False


def model_refresh_thread():
    """Thread function to periodically check for model updates"""
    while True:
        try:
            check_and_reload_model()
            time.sleep(120)  # Check every 2 minutes
        except Exception as e:
            logger.error(f"Error in model refresh thread: {e}")
            logger.error(traceback.format_exc())
            time.sleep(60)  # Wait a bit before retrying


def load_latest_data():
    """Load the latest data for predictions"""
    try:
        # Check if data directory exists
        if not os.path.exists(DATA_DIR):
            logger.warning("Data directory does not exist, creating it")
            os.makedirs(DATA_DIR)
            return None, None

        # Find the latest data files
        trends_file = None
        driver_file = None
        funnel_file = None

        for file in os.listdir(DATA_DIR):
            if file.endswith(".json"):
                file_path = os.path.join(DATA_DIR, file)

                if "trends" in file.lower():
                    trends_file = file_path
                elif "driver" in file.lower():
                    driver_file = file_path
                elif "funnel" in file.lower():
                    funnel_file = file_path

        if not trends_file or not driver_file:
            logger.error("Missing required data files")
            logger.error(f"Found files: {os.listdir(DATA_DIR)}")
            return None, None

        logger.info(f"Loading data from: {trends_file}, {driver_file}, {funnel_file}")

        # Load the data
        trends_df = pd.read_json(trends_file)
        driver_df = pd.read_json(driver_file)

        if funnel_file:
            funnel_df = pd.read_json(funnel_file)

            # Merge the data
            merged_df = pd.merge(trends_df, driver_df, on=["ward_num"], how="left")
            merged_df = pd.merge(
                merged_df, funnel_df, on=["ward_num", "vehicle_type"], how="left"
            )

            # Handle duplicate columns
            duplicate_cols = [
                col
                for col in merged_df.columns
                if col.endswith("_x") or col.endswith("_y")
            ]
            for col in duplicate_cols:
                base_col = col[:-2]
                if col.endswith("_x"):
                    merged_df.drop(columns=[col], inplace=True)
                else:
                    merged_df.rename(columns={col: base_col}, inplace=True)
        else:
            merged_df = pd.merge(trends_df, driver_df, on=["ward_num"], how="left")

        # Fill NaN values
        for col in merged_df.columns:
            if col not in ["ward_num", "vehicle_type", "date"]:
                merged_df[col] = merged_df[col].fillna(0)

        # Calculate important metrics if they don't exist
        if "earning" in merged_df.columns and "done_ride" in merged_df.columns:
            # Calculate avg_earning_per_ride
            merged_df["avg_earning_per_ride"] = merged_df.apply(
                lambda row: (
                    row["earning"] / row["done_ride"] if row["done_ride"] > 0 else 0
                ),
                axis=1,
            )
            logger.info(
                "Calculated avg_earning_per_ride from earnings and done_ride data"
            )

            # Log some statistics about the earnings data
            logger.info(f"Earnings data statistics:")
            logger.info(
                f"Min: {merged_df['earning'].min()}, Max: {merged_df['earning'].max()}, Mean: {merged_df['earning'].mean()}"
            )
            logger.info(f"Done rides statistics:")
            logger.info(
                f"Min: {merged_df['done_ride'].min()}, Max: {merged_df['done_ride'].max()}, Mean: {merged_df['done_ride'].mean()}"
            )
            logger.info(f"Avg earning per ride statistics:")
            logger.info(
                f"Min: {merged_df['avg_earning_per_ride'].min()}, Max: {merged_df['avg_earning_per_ride'].max()}, Mean: {merged_df['avg_earning_per_ride'].mean()}"
            )
        else:
            logger.warning(
                "Cannot calculate avg_earning_per_ride: missing 'earning' or 'done_ride' columns"
            )

        # Calculate conversion_rate if it doesn't exist
        if (
            "booking" in merged_df.columns
            and "srch_rqst" in merged_df.columns
            and "conversion_rate" not in merged_df.columns
        ):
            merged_df["conversion_rate"] = merged_df.apply(
                lambda row: (
                    row["booking"] / row["srch_rqst"] if row["srch_rqst"] > 0 else 0
                ),
                axis=1,
            )

        # Calculate completion_rate if it doesn't exist
        if (
            "done_ride" in merged_df.columns
            and "booking" in merged_df.columns
            and "completion_rate" not in merged_df.columns
        ):
            merged_df["completion_rate"] = merged_df.apply(
                lambda row: (
                    row["done_ride"] / row["booking"] if row["booking"] > 0 else 0
                ),
                axis=1,
            )

        # Calculate cancellation_rate if it doesn't exist
        if (
            "cancel_ride" in merged_df.columns
            and "booking" in merged_df.columns
            and "cancellation_rate" not in merged_df.columns
        ):
            merged_df["cancellation_rate"] = merged_df.apply(
                lambda row: (
                    row["cancel_ride"] / row["booking"] if row["booking"] > 0 else 0
                ),
                axis=1,
            )

        # Calculate rider_satisfaction_proxy if it doesn't exist
        if (
            "rider_cancel" in merged_df.columns
            and "booking" in merged_df.columns
            and "rider_satisfaction_proxy" not in merged_df.columns
        ):
            merged_df["rider_satisfaction"] = merged_df.apply(
                lambda row: (
                    1 - (row["rider_cancel"] / row["booking"])
                    if row["booking"] > 0
                    else 0.5
                ),
                axis=1,
            )

        logger.info(f"Data loaded successfully with {len(merged_df)} rows")
        logger.info(f"Columns available: {merged_df.columns.tolist()}")
        return merged_df, merged_df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.error(traceback.format_exc())
        return None, None


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    if model is not None and features is not None:
        return jsonify(
            {
                "status": "healthy",
                "model_loaded": True,
                "model_info": model_info,
                "last_loaded": (
                    last_model_load_time.isoformat() if last_model_load_time else None
                ),
            }
        )
    else:
        return (
            jsonify(
                {
                    "status": "unhealthy",
                    "model_loaded": False,
                    "error": "Model not loaded",
                }
            ),
            503,
        )


@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None or features is None:
            return jsonify({"error": "Model not loaded"}), 503

        # Get request data
        data = request.json

        # Validate required fields
        required_fields = ["user_time", "user_location", "user_vehicle_type"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Load the latest data
        df, original_df = load_latest_data()
        if df is None:
            return jsonify({"error": "Failed to load data"}), 500

        # Get all wards
        all_wards = sorted(original_df["ward_num"].unique())

        # Make predictions
        recommendations = predict_and_recommend(
            model,
            features,
            data["user_time"],
            data["user_location"],
            data["user_vehicle_type"],
            all_wards,
            df,
            original_df,
        )

        # Process recommendations for response
        for rec in recommendations:
            # Add explanation about distance impact
            if "distance_factor" in rec:
                rec["distance_explanation"] = (
                    f"Distance factor: {rec['distance_factor']:.2f}. "
                    f"Closer wards receive higher scores, especially during peak hours."
                )
                # Remove the raw factor from the response
                del rec["distance_factor"]

            # Format demand ratio for display
            if "demand_ratio" in rec:
                rec["demand_ratio_explanation"] = (
                    f"This area has {rec['demand_ratio']:.1f}x the average demand "
                    f"compared to nearby areas."
                )

            # Add detailed metrics for each recommendation
            rec["detailed_metrics"] = {
                "earnings": {
                    "avg_per_ride": rec.get("avg_earning_per_ride", 0),
                    "avg_fare": rec.get("avg_fare", 0),
                    "estimated_hourly": rec.get("estimated_hourly_earnings", 0),
                    "estimated_rides_per_hour": rec.get("estimated_rides_per_hour", 0),
                    "currency": "INR",
                },
                "ride_metrics": {
                    "conversion_rate": rec.get("conversion_rate", 0),
                    "completion_rate": rec.get("completion_rate", 0),
                    "cancellation_rate": rec.get("cancellation_rate", 0),
                    "avg_distance_per_trip": rec.get("avg_distance_per_trip", 0),
                    "rider_satisfaction": rec.get("rider_satisfaction", 0),
                },
                "demand_metrics": {
                    "predicted_demand": rec.get("predicted_demand", 0),
                    "search_requests": rec.get("search_requests", 0),
                    "demand_ratio": rec.get("demand_ratio", 1.0),
                    "current_drivers": rec.get("current_drivers", 0),
                },
                "time_factors": {
                    "is_peak_hour": rec.get("is_peak_hour", 0),
                    "time_of_day": rec.get("time_of_day", "Unknown"),
                    "is_weekend": rec.get("is_weekend", 0),
                    "is_holiday": rec.get("is_holiday", 0),
                },
            }

            # Ensure all recommendations have proper earnings data
            if rec.get("avg_earning_per_ride", 0) <= 0:
                # Don't set default values, just log a warning
                logger.warning(f"Ward {rec['ward']} has no earnings data")

            # Add earnings potential explanation
            estimated_hourly = rec.get("estimated_hourly_earnings", 0)
            rec["earnings_potential_explanation"] = (
                f"Estimated hourly earnings of ₹{estimated_hourly:.2f} based on "
                f"approximately {rec.get('estimated_rides_per_hour', 0):.1f} rides per hour "
                f"with an average of ₹{rec.get('avg_earning_per_ride', 0):.2f} per ride."
            )

            # Add time-based explanation
            time_of_day = rec.get("time_of_day", "Unknown")
            is_peak = rec.get("is_peak_hour", 0)
            is_weekend = rec.get("is_weekend", 0)
            is_holiday = rec.get("is_holiday", 0)

            time_explanation = f"Current time is {time_of_day.lower()}"
            if is_peak:
                time_explanation += " during peak hours"
            if is_weekend:
                time_explanation += " on a weekend"
            elif is_holiday:
                time_explanation += " on a holiday"

            rec["time_explanation"] = time_explanation + "."

            # Add rider satisfaction explanation
            if "rider_satisfaction" in rec:
                satisfaction_level = (
                    "high"
                    if rec["rider_satisfaction"] > 0.8
                    else "moderate" if rec["rider_satisfaction"] > 0.6 else "low"
                )
                rec["rider_satisfaction_explanation"] = (
                    f"This area has {satisfaction_level} rider satisfaction ({rec['rider_satisfaction']:.2f}). "
                    f"Higher values indicate fewer rider cancellations."
                )

            # Add queue effectiveness explanation
            if "queue_effectiveness" in rec:
                rec["queue_effectiveness_explanation"] = (
                    f"Queue effectiveness of {rec['queue_effectiveness']:.2f} indicates how well "
                    f"the queue system matches drivers to ride requests in this area."
                )

            # Remove internal fields that shouldn't be in the API response
            if "demand_boost" in rec:
                del rec["demand_boost"]

        # Check if all recommendations are far away
        all_far = all(rec["distance"] > 15 for rec in recommendations)

        # Add a general explanation if all recommendations are far away
        general_explanation = ""
        if all_far:
            general_explanation = (
                "Note: All recommended areas are relatively far from your location. "
                "This is because nearby areas currently have low predicted demand. "
                "Consider these recommendations for higher earning potential despite the distance."
            )

        # Format response
        response = {
            "recommendations": recommendations,
            "model_info": {
                "trained_at": model_info.get("trained_at"),
                "num_features": model_info.get("num_features"),
            },
            "timestamp": datetime.now().isoformat(),
            "explanation": {
                "scoring_factors": {
                    "demand": "25% - Predicted demand based on historical patterns",
                    "search_requests": "15% - Recent search activity in the ward",
                    "distance": "25% - Proximity to your current location (higher weight during peak hours)",
                    "drivers": "10% - Current driver availability",
                    "conversion": "15% - How often searches convert to bookings",
                    "cancellation": "10% - How often rides are completed without cancellation",
                }
            },
            "earnings_info": {
                "calculation_method": "Average earnings are calculated based on historical completed rides in each area",
                "estimated_hourly_calculation": "Estimated hourly earnings = Average earnings per ride × Estimated rides per hour",
                "estimated_rides_calculation": "Estimated rides per hour is based on historical data for the current time of day",
                "peak_hour_impact": "During peak hours (7-9 AM, 5-7 PM), ride frequency and fares may be higher",
                "vehicle_type_impact": f"Calculations are specific to your selected vehicle type: {data['user_vehicle_type']}",
                "factors_affecting_earnings": [
                    "Ride distance - Longer rides typically yield higher earnings",
                    "Time of day - Peak hours generally have higher fares",
                    "Vehicle type - Different vehicle types have different base fares",
                    "Demand level - Higher demand may lead to surge pricing",
                    "Cancellation rate - High cancellations reduce effective hourly earnings",
                    "Traffic conditions - Affects how many rides can be completed per hour",
                ],
            },
        }

        # Add general explanation if needed
        if general_explanation:
            response["general_explanation"] = general_explanation

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        logger.error(traceback.format_exc())

        return jsonify({"error": str(e)}), 500


def main():
    """Main function to start the API server"""
    try:
        logger.info("Starting API server")

        # Check if data directory exists
        if not os.path.exists(DATA_DIR):
            logger.warning("Data directory does not exist, creating it")
            os.makedirs(DATA_DIR)

        # Check if models directory exists
        if not os.path.exists("models"):
            logger.warning("Models directory does not exist, creating it")
            os.makedirs("models")

        # Load the model
        if not load_model():
            logger.warning(
                "Failed to load model, will retry when model becomes available"
            )

        # Start model refresh thread
        refresh_thread = threading.Thread(target=model_refresh_thread, daemon=True)
        refresh_thread.start()

        # Start the Flask app
        logger.info("Starting Flask server on port 8000")
        app.run(host="0.0.0.0", port=8000, debug=False)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
