import json
import os
import pickle
import traceback
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from custom_logger import CustomLogger
import schedule
import time
from threading import Thread
from data_manager import RideDataManager


logger = CustomLogger("forecaster", "logs/forecaster_log.txt")


class RideRequestForecast:
    """Class to forecast ride requests based on historical data"""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.model = None
        self.features = []
        self.last_training_time = None
        self.forecast_history = []

    def prepare_data_for_modeling(self, data_type="ward"):
        """Prepare data for modeling"""
        try:
            df = self.data_manager.get_historical_trends_data(data_type)
            if df.empty:
                logger.warning("No data available for modeling")
                return None

            if "datetime" not in df.columns:
                logger.error("Missing 'datetime' column in data")
                return None

            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

            if df["datetime"].isna().any():
                logger.warning(
                    "Invalid datetime values found in data. Replacing with current time."
                )
                df["datetime"].fillna(datetime.now(), inplace=True)

            if "srch_rqst" not in df.columns:
                logger.error("Missing 'srch_rqst' column in data")
                return None

            region_col = "ward_num" if data_type == "ward" else "ac_num"
            if region_col not in df.columns:
                logger.error(f"Missing region column: {region_col}")
                return None

            df["hour_of_day"] = df["datetime"].dt.hour
            df["day_of_week"] = df["datetime"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
            df["is_morning_rush"] = df["hour_of_day"].apply(
                lambda x: 1 if 7 <= x <= 10 else 0
            )
            df["is_evening_rush"] = df["hour_of_day"].apply(
                lambda x: 1 if 16 <= x <= 19 else 0
            )

            agg_df = df.groupby([region_col, "datetime"], as_index=False).agg(
                {
                    "srch_rqst": "sum",
                    "hour_of_day": "first",
                    "day_of_week": "first",
                    "is_weekend": "first",
                    "is_morning_rush": "first",
                    "is_evening_rush": "first",
                }
            )

            agg_df[region_col] = agg_df[region_col].astype(str)
            agg_df = agg_df.sort_values([region_col, "datetime"])

            for lag in range(1, 25):
                agg_df[f"lag_{lag}"] = agg_df.groupby(region_col)["srch_rqst"].shift(
                    lag
                )

            for window in [3, 6, 12, 24]:
                agg_df[f"rolling_mean_{window}"] = agg_df.groupby(region_col)[
                    "srch_rqst"
                ].transform(lambda x: x.rolling(window, min_periods=1).mean())

            agg_df = agg_df.fillna(0)
            logger.info(f"Prepared data for modeling with {len(agg_df)} records")
            return agg_df

        except Exception as e:
            logger.error(f"Error preparing data for modeling: {e}")
            return None

    def train_model(self, data_type="ward", force=False):
        """Train the forecasting model"""
        try:
            if (
                not force
                and self.model is not None
                and self.last_training_time is not None
            ):
                logger.info("Model already trained and force is not set")
                return self.model

            df = self.prepare_data_for_modeling(data_type)

            if df is None or df.empty:
                logger.error("No data available for training")
                return None

            feature_cols = [
                col
                for col in df.columns
                if col.startswith("lag_")
                or col.startswith("rolling_mean_")
                or col
                in [
                    "hour_of_day",
                    "day_of_week",
                    "is_weekend",
                    "is_morning_rush",
                    "is_evening_rush",
                ]
            ]

            if len(df) < 2 * len(feature_cols):
                logger.error("Insufficient data for training")
                return None

            self.features = feature_cols

            modeling_df = df.dropna(subset=feature_cols + ["srch_rqst"])

            if len(modeling_df) < 10:
                logger.error("Insufficient data points after dropping NA")
                return None

            X = modeling_df[feature_cols]
            y = modeling_df["srch_rqst"]

            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=min(10, len(feature_cols)),
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            )
            self.model.fit(X, y)

            
            os.makedirs("cache/models", exist_ok=True)

            model_path = os.path.join("cache/models", f"rf_model_{data_type}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)

            features_path = os.path.join("cache/models", f"features_{data_type}.json")
            with open(features_path, "w") as f:
                json.dump(self.features, f, indent=2)

            self.last_training_time = datetime.now()

            feature_importance = pd.DataFrame(
                {"Feature": X.columns, "Importance": self.model.feature_importances_}
            ).sort_values("Importance", ascending=False)

            logger.info(
                f"Model trained successfully with {len(X)} data points and {len(feature_cols)} features"
            )
            logger.info(
                f"\nTop 10 most important features:\n{feature_importance.head(10)}"
            )
            logger.debug(f"Feature importance: {feature_importance}")

            return self.model
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None

    def load_model(self, data_type="ward"):
        """Load a pre-trained model from disk."""
        try:
            model_path = os.path.join("cache/models", f"rf_model_{data_type}.pkl")
            features_path = os.path.join("cache/models", f"features_{data_type}.json")

            if os.path.exists(model_path) and os.path.exists(features_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)

                with open(features_path, "r") as f:
                    self.features = json.load(f)

                logger.info(f"Model and features loaded successfully for {data_type}.")
                return True
            else:
                logger.warning(f"Model or features not found for {data_type}.")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def rebuild_model_if_needed(self, data_type="ward"):
        """Rebuild the model if the last training time is more than 24 hours ago."""
        try:
            if (
                self.last_training_time is None
                or (datetime.now() - self.last_training_time).total_seconds() > 86400
            ):
                logger.info("Model is outdated or not trained. Rebuilding the model...")
                self.train_model(data_type=data_type, force=True)
            else:
                logger.info("Model is up-to-date. No need to rebuild.")
        except Exception as e:
            logger.error(f"Error checking or rebuilding the model: {e}")

    def forecast_next_hours(self, hours=24, data_type="ward", region=None):
        """Forecast ride requests for the next specified hours"""
        try:
            if self.model is None:
                logger.error("Model is not trained")
                return None

            df = self.prepare_data_for_modeling(data_type)

            if df is None or df.empty:
                logger.error("No data available for forecasting")
                return None

            region_col = "ward_num" if data_type == "ward" else "ac_num"

            if region is not None:
                df = df[df[region_col] == region]

            if region_col not in df.columns:
                logger.error(f"Missing region column: {region_col} in prepared data")
                return None

            forecasts = []

            for curr_region in df[region_col].unique():
                region_df = df[df[region_col] == curr_region]
                if region_df.empty:
                    continue

                X = region_df[self.features].tail(hours)
                if X.empty:
                    logger.warning(
                        f"Insufficient data for region {curr_region} to forecast {hours} hours"
                    )
                    continue

                predictions = self.model.predict(X)

                for i, pred in enumerate(predictions):
                    forecast_time = datetime.now() + timedelta(hours=i)
                    if not isinstance(forecast_time, datetime):
                        logger.error("Forecast time is not a datetime object")
                        continue
                    forecasts.append(
                        {
                            "datetime": forecast_time.isoformat(),
                            region_col: curr_region,
                            "forecast_requests": int(pred),
                        }
                    )

            if not forecasts:
                logger.warning("No forecasts generated")
                return None

            forecast_df = pd.DataFrame(forecasts)

            if "ward_num" not in forecast_df.columns and data_type == "ward":
                logger.error("Generated forecast is missing 'ward_num' column")
                return None

            logger.info(
                f"Forecast generated for {hours} hours: {len(forecast_df)} records"
            )
            return forecast_df
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            logger.error(traceback.format_exc())
            return None

    def fetch_and_prepare_data(self):
        """Fetch data dynamically and prepare it for modeling."""
        try:
            logger.info("Fetching and preparing data dynamically...")
            self.data_manager.fetch_all_endpoints()
            logger.info("Data fetched and prepared successfully.")
        except Exception as e:
            logger.error(f"Error fetching and preparing data: {e}")

    def train_and_save_model(self):
        """Train the model and save it to disk."""
        try:
            logger.info("Training and saving the model...")
            self.train_model(force=True)
            logger.info("Model trained and saved successfully.")
        except Exception as e:
            logger.error(f"Error training and saving the model: {e}")

    def schedule_training(self):
        """Schedule periodic training of the model."""

        def training_job():
            logger.info("Scheduled training job started.")
            self.fetch_and_prepare_data()
            self.train_and_save_model()

        schedule.every(720).minutes.do(training_job)
        logger.info("Scheduled training every 720 minutes.")

        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(1)

        Thread(target=run_scheduler, daemon=True).start()



if __name__ == "__main__":
    data_manager = RideDataManager()
    forecaster = RideRequestForecast(data_manager)
    forecaster.schedule_training()
    logger.info("Forecaster is running with scheduled training.")

    
    while True:
        time.sleep(10)
