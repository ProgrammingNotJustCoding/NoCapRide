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
from model import load_and_preprocess_data, create_features, predict_earnings

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

# Create a separate handler for detailed output
detailed_logger = logging.getLogger("train_detailed")
detailed_logger.setLevel(logging.INFO)
detailed_handler = logging.FileHandler("logs/serve_output.log", mode="a")
detailed_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
detailed_logger.addHandler(detailed_handler)
detailed_logger.propagate = False  # Prevent double logging

logger = logging.getLogger("serve")

# Global variables
MODEL_FILE = "models/demand_model.pkl"
MODEL_INFO_FILE = "models/model_info.json"
DATA_REFRESH_INTERVAL = 60  # seconds

# Global data and model objects
models = None
model_info = None
data = None
data_lock = threading.Lock()

app = Flask(__name__)


def load_model():
    """Load the trained model from disk"""
    global models, model_info
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(MODEL_INFO_FILE):
            with open(MODEL_FILE, "rb") as f:
                models = pickle.load(f)

            with open(MODEL_INFO_FILE, "r") as f:
                model_info = json.load(f)

            logger.info(
                f"Model loaded successfully. Trained at: {model_info['trained_at']}"
            )
            return True
        else:
            logger.error("Model files not found")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        return False


def load_data():
    """Load and preprocess the latest data"""
    global data
    try:
        # Find the latest data files
        data_dir = "data"
        trends_file = None
        driver_file = None
        funnel_file = None

        for file in os.listdir(data_dir):
            if file.endswith(".json"):
                file_path = os.path.join(data_dir, file)

                if "trends" in file.lower():
                    trends_file = file_path
                elif "driver" in file.lower():
                    driver_file = file_path
                elif "funnel" in file.lower():
                    funnel_file = file_path

        if not trends_file or not driver_file:
            logger.error("Missing required data files")
            return False

        logger.info(f"Loading data from: {trends_file}, {driver_file}, {funnel_file}")

        # Load and preprocess the data
        merged_df = load_and_preprocess_data(trends_file, driver_file, funnel_file)

        if merged_df is None:
            logger.error("Failed to load and preprocess data")
            return False

        # Calculate avg_earning_per_ride if not already present
        if (
            "earning" in merged_df.columns
            and "done_ride" in merged_df.columns
            and "avg_earning_per_ride" not in merged_df.columns
        ):
            merged_df["avg_earning_per_ride"] = merged_df.apply(
                lambda row: (
                    row["earning"] / row["done_ride"] if row["done_ride"] > 0 else 0
                ),
                axis=1,
            )
            logger.info(
                "Calculated avg_earning_per_ride from earnings and done_ride data"
            )

            # Log earnings statistics
            logger.info("Earnings data statistics:")
            logger.info(
                f"Min: {merged_df['earning'].min()}, Max: {merged_df['earning'].max()}, Mean: {merged_df['earning'].mean()}"
            )

            # Log done_rides statistics
            logger.info("Done rides statistics:")
            logger.info(
                f"Min: {merged_df['done_ride'].min()}, Max: {merged_df['done_ride'].max()}, Mean: {merged_df['done_ride'].mean()}"
            )

            # Log avg_earning_per_ride statistics
            logger.info("Avg earning per ride statistics:")
            logger.info(
                f"Min: {merged_df['avg_earning_per_ride'].min()}, Max: {merged_df['avg_earning_per_ride'].max()}, Mean: {merged_df['avg_earning_per_ride'].mean()}"
            )

        # Create features for prediction
        feature_df = create_features(merged_df)

        if feature_df is None:
            logger.error("Failed to create features")
            return False

        # Update the global data with a lock to prevent race conditions
        with data_lock:
            data = {
                "merged_df": merged_df,
                "feature_df": feature_df,
                "last_updated": datetime.now(),
            }

        logger.info(f"Data loaded successfully with {len(merged_df)} rows")
        logger.info(f"Columns available: {merged_df.columns.tolist()}")
        return True

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.error(traceback.format_exc())
        return False


def data_refresh_thread():
    """Thread function to periodically refresh the data"""
    while True:
        try:
            load_data()
            time.sleep(DATA_REFRESH_INTERVAL)
        except Exception as e:
            logger.error(f"Error in data refresh thread: {e}")
            logger.error(traceback.format_exc())
            time.sleep(10)  # Wait a bit before retrying


def get_ward_recommendations(user_time, user_location, user_vehicle_type):
    """Get recommendations for the best wards based on earnings potential"""
    global models, data

    try:
        if models is None or data is None:
            return {"error": "Model or data not loaded"}, 500

        # Get the merged data and feature data
        merged_df = data["merged_df"]
        feature_df = data["feature_df"]

        # Convert user_time to datetime
        user_time_dt = pd.to_datetime(user_time)

        # Filter data for the specified hour
        hour_data = feature_df[feature_df["hour"] == user_time_dt.hour].copy()

        # If vehicle type is specified, filter for that vehicle type
        if user_vehicle_type and "vehicle_type" in merged_df.columns:
            vehicle_data = hour_data[
                hour_data["vehicle_type"] == user_vehicle_type
            ].copy()
            if not vehicle_data.empty:
                hour_data = vehicle_data

        # If no data for the specified hour, use all data
        if hour_data.empty:
            hour_data = feature_df.copy()

        # Make predictions using the best model
        try:
            best_model_name = models["best_model"]
            predictions_df = predict_earnings(
                models, hour_data, model_name=best_model_name
            )
        except (TypeError, KeyError) as e:
            logger.warning(
                f"Error accessing best model: {e}. Using random forest model instead."
            )
            # Fall back to using the random forest model
            predictions_df = predict_earnings(
                models, hour_data, model_name="random_forest"
            )

        if predictions_df is None:
            return {"error": "Failed to make predictions"}, 500

        # Get the ward numbers
        wards = merged_df["ward_num"].unique()

        # Calculate distances from user_location
        try:
            user_ward = int(user_location)
            predictions_df["distance"] = predictions_df["ward_num"].apply(
                lambda ward: abs(int(ward) - user_ward)
            )
        except:
            # If user_location is not a valid ward number, set all distances to 0
            predictions_df["distance"] = 0

        # Calculate a score based on predicted earnings and distance
        predictions_df["score"] = 0.7 * predictions_df[
            "predicted_earnings"
        ] / predictions_df["predicted_earnings"].max() - 0.3 * predictions_df[
            "distance"
        ] / (
            predictions_df["distance"].max()
            if predictions_df["distance"].max() > 0
            else 1
        )

        # Sort by score in descending order
        predictions_df = predictions_df.sort_values("score", ascending=False)

        # Get the top 5 wards
        top_wards = predictions_df.head(5)

        # Check if any of the top wards have earnings data
        for ward in top_wards["ward_num"].unique():
            ward_data = merged_df[merged_df["ward_num"] == ward]
            if "earning" not in ward_data.columns or ward_data["earning"].sum() == 0:
                logger.warning(f"Ward {ward} has no earnings data")

        # Prepare the response
        recommendations = []
        for _, row in top_wards.iterrows():
            ward = row["ward_num"]
            ward_data = merged_df[merged_df["ward_num"] == ward]

            # Get additional metrics for this ward
            metrics = {
                "ward": ward,
                "predicted_earnings": float(row["predicted_earnings"]),
                "distance": int(row["distance"]),
                "score": float(row["score"]),
            }

            # Add additional metrics if available
            if "active_drvr" in ward_data.columns:
                metrics["active_drivers"] = int(ward_data["active_drvr"].mean())

            if "srch_rqst" in ward_data.columns:
                metrics["search_requests"] = int(ward_data["srch_rqst"].mean())

            if "conversion_rate" in ward_data.columns:
                metrics["conversion_rate"] = float(ward_data["conversion_rate"].mean())

            if "completion_rate" in ward_data.columns:
                metrics["completion_rate"] = float(ward_data["completion_rate"].mean())

            if "cancellation_rate" in ward_data.columns:
                metrics["cancellation_rate"] = float(
                    ward_data["cancellation_rate"].mean()
                )

            if "avg_earning_per_ride" in ward_data.columns:
                metrics["avg_earning_per_ride"] = float(
                    ward_data["avg_earning_per_ride"].mean()
                )

            if "rider_satisfaction_proxy" in ward_data.columns:
                metrics["rider_satisfaction"] = float(
                    ward_data["rider_satisfaction_proxy"].mean()
                )

            recommendations.append(metrics)

        return {"recommendations": recommendations}, 200

    except Exception as e:
        logger.error(f"Error getting ward recommendations: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}, 500


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to get predictions"""
    try:
        # Get request data
        request_data = request.get_json()

        # Check required parameters
        if not request_data:
            return jsonify({"error": "No data provided"}), 400

        # Get parameters
        user_time = request_data.get("user_time", datetime.now().isoformat())
        user_location = request_data.get("user_location", "")
        user_vehicle_type = request_data.get("user_vehicle_type", "")

        # Get recommendations
        recommendations, status_code = get_ward_recommendations(
            user_time, user_location, user_vehicle_type
        )

        return jsonify(recommendations), status_code

    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """API endpoint to check the health of the service"""
    global models, data

    status = {
        "status": "healthy",
        "model_loaded": models is not None,
        "data_loaded": data is not None,
    }

    if models is not None and "best_model" in models:
        status["best_model"] = models["best_model"]

    if data is not None and "last_updated" in data:
        status["data_last_updated"] = data["last_updated"].isoformat()

    if model_info is not None and "trained_at" in model_info:
        status["model_trained_at"] = model_info["trained_at"]

    return jsonify(status), 200


def main():
    """Main function to start the API server"""
    logger.info("Starting API server")

    # Load the model
    if not load_model():
        logger.error("Failed to load model, exiting")
        sys.exit(1)

    # Load initial data
    if not load_data():
        logger.warning("Failed to load initial data, will retry in background")

    # Start data refresh thread
    refresh_thread = threading.Thread(target=data_refresh_thread, daemon=True)
    refresh_thread.start()

    # Start Flask server
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
