import os
import time
import json
import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import schedule
from model import load_and_preprocess_data, create_features, train_models

# Ensure the logs directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Set up logging with increased buffer size and proper formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/train_log.txt", mode="a"),
        logging.StreamHandler(),
    ],
)

# Create a separate handler for detailed output
detailed_logger = logging.getLogger("train_detailed")
detailed_logger.setLevel(logging.INFO)
detailed_handler = logging.FileHandler("logs/train_output.log", mode="a")
detailed_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
detailed_logger.addHandler(detailed_handler)
detailed_logger.propagate = False  # Prevent double logging

logger = logging.getLogger("train")

# Ensure the models directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# Global variables
MODEL_FILE = "models/demand_model.pkl"
FEATURES_FILE = "models/model_features.pkl"
MODEL_INFO_FILE = "models/model_info.json"


def find_latest_data_files():
    """Find the latest data files in the data directory"""
    data_dir = "data"
    trends_file = None
    driver_file = None
    funnel_file = None

    # Look for the most recent files
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            file_path = os.path.join(data_dir, file)

            # Check if it's one of our target files
            if "trends" in file.lower():
                trends_file = file_path
            elif "driver" in file.lower():
                driver_file = file_path
            elif "funnel" in file.lower():
                funnel_file = file_path

    return trends_file, driver_file, funnel_file


def train_and_save_model():
    """Train the model and save it to disk"""
    try:
        logger.info("Starting model training")

        # Find the latest data files
        trends_file, driver_file, funnel_file = find_latest_data_files()

        if not trends_file or not driver_file:
            logger.error("Missing required data files")
            return False

        logger.info(f"Using data files: {trends_file}, {driver_file}, {funnel_file}")

        # Load and preprocess the data
        merged_df = load_and_preprocess_data(trends_file, driver_file, funnel_file)

        if merged_df is None:
            logger.error("Failed to load and preprocess data")
            return False

        # Create features
        df_with_features = create_features(merged_df)

        if df_with_features is None:
            logger.error("Failed to create features")
            return False

        # Train models (both RandomForest and GradientBoosting)
        trained_models = train_models(
            df_with_features, target_variable="avg_earning_per_ride"
        )

        if trained_models is None:
            logger.error("Failed to train models")
            return False

        # Save the models and features
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(trained_models, f)

        # Save model metadata
        model_info = {
            "trained_at": datetime.now().isoformat(),
            "trends_file": trends_file,
            "driver_file": driver_file,
            "funnel_file": funnel_file,
            "best_model": trained_models["best_model"],
            "random_forest": {
                "val_rmse": float(trained_models["random_forest"]["val_rmse"]),
                "val_r2": float(trained_models["random_forest"]["val_r2"]),
                "test_rmse": float(trained_models["random_forest"]["test_rmse"]),
                "test_r2": float(trained_models["random_forest"]["test_r2"]),
            },
            "gradient_boosting": {
                "val_rmse": float(trained_models["gradient_boosting"]["val_rmse"]),
                "val_r2": float(trained_models["gradient_boosting"]["val_r2"]),
                "test_rmse": float(trained_models["gradient_boosting"]["test_rmse"]),
                "test_r2": float(trained_models["gradient_boosting"]["test_r2"]),
            },
            "num_features": len(trained_models["random_forest"]["features"]),
            "num_samples": len(df_with_features),
            "wards": merged_df["ward_num"].unique().tolist(),
            "vehicle_types": (
                merged_df["vehicle_type"].unique().tolist()
                if "vehicle_type" in merged_df.columns
                else []
            ),
        }

        with open(MODEL_INFO_FILE, "w") as f:
            json.dump(model_info, f, indent=2)

        logger.info(
            f"Model trained and saved successfully with {model_info['num_features']} features"
        )
        return True

    except Exception as e:
        logger.error(f"Error training model: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def scheduled_training():
    """Function to be called by the scheduler"""
    logger.info("Running scheduled model training")
    success = train_and_save_model()
    if success:
        logger.info("Scheduled training completed successfully")
    else:
        logger.error("Scheduled training failed")


def main():
    """Main function to run the continuous training process"""
    logger.info("Starting training service")

    # Train the model immediately on startup
    train_and_save_model()

    # Schedule training every 2 minutes
    schedule.every(2).minutes.do(scheduled_training)

    logger.info("Training service started, will train model every 2 minutes")

    # Run the scheduling loop
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
