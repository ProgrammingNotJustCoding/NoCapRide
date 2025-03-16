import os
import requests
import json
import logging
import traceback
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import schedule
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/forecast_log.txt", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("forecast")

# Create a separate logger for detailed output
detailed_logger = logging.getLogger("forecast_detailed")
detailed_logger.setLevel(logging.INFO)
detailed_handler = logging.FileHandler("logs/forecast_output.log", mode="a")
detailed_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
detailed_logger.addHandler(detailed_handler)
detailed_logger.propagate = False  # Prevent double logging

# Ensure necessary directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("data/historical", exist_ok=True)
os.makedirs("data/models", exist_ok=True)
os.makedirs("data/forecasts", exist_ok=True)
os.makedirs("data/visualizations", exist_ok=True)

# Base URL for the CDN
BASE_URL = "https://d11gklsvr97l1g.cloudfront.net/open/json-data"

# List of endpoints to fetch
ENDPOINTS = [
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


class RideDataManager:
    """Class to manage ride data fetching, storage, and retrieval"""

    def __init__(self):
        self.last_fetch_time = None
        self.historical_data = {}

    def fetch_endpoint(self, endpoint):
        """Fetch data from a specific endpoint"""
        url = f"{BASE_URL}/{endpoint}"
        try:
            logger.info(f"Fetching data from {url}")
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                # Save to latest data file
                file_path = os.path.join("data", endpoint)
                with open(file_path, "w") as f:
                    f.write(response.text)

                # Parse the data
                data = json.loads(response.text)

                # Save to historical data with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                hist_file_path = os.path.join(
                    "data/historical", f"{timestamp}_{endpoint}"
                )
                with open(hist_file_path, "w") as f:
                    f.write(response.text)

                logger.info(f"Successfully fetched and saved {endpoint}")
                return data
            else:
                logger.error(
                    f"Failed to fetch {endpoint}. Status code: {response.status_code}"
                )
                return self._load_latest_data(endpoint)
        except Exception as e:
            logger.error(f"Error fetching {endpoint}: {e}")
            logger.error(traceback.format_exc())
            return self._load_latest_data(endpoint)

    def _load_latest_data(self, endpoint):
        """Load the latest data file if available as fallback"""
        file_path = os.path.join("data", endpoint)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                logger.info(f"Loaded existing {endpoint} as fallback")
                return data
            except Exception as e:
                logger.error(f"Error loading fallback data for {endpoint}: {e}")
        return None

    def fetch_all_endpoints(self):
        """Fetch data from all endpoints"""
        fetched_data = {}
        fetch_success = False

        for endpoint in ENDPOINTS:
            data = self.fetch_endpoint(endpoint)
            if data is not None:
                fetched_data[endpoint] = data
                fetch_success = True

        if fetch_success:
            self.last_fetch_time = datetime.now()
            # Update the historical dataset
            self.update_historical_dataset(fetched_data)

        return fetched_data

    def update_historical_dataset(self, new_data):
        """Update the historical dataset with new data"""
        try:
            # Process and store trends data
            if "trends_live_ward_new_key.json" in new_data:
                self._process_trends_data(
                    new_data["trends_live_ward_new_key.json"], "ward"
                )

            if "trends_live_ca_new_key.json" in new_data:
                self._process_trends_data(new_data["trends_live_ca_new_key.json"], "ca")

            # Save the updated historical data
            self._save_historical_data()

            logger.info("Historical dataset updated successfully")
        except Exception as e:
            logger.error(f"Error updating historical dataset: {e}")
            logger.error(traceback.format_exc())

    def _process_trends_data(self, data, data_type):
        """Process trends data and add to historical collection"""
        try:
            # Initialize historical structure if needed
            if f"trends_{data_type}" not in self.historical_data:
                self.historical_data[f"trends_{data_type}"] = []

            # Ensure data is in list format
            if isinstance(data, dict):
                data = [data]

            # Add current timestamp if not present
            current_time = datetime.now()
            for item in data:
                if "date" not in item or not item["date"]:
                    item["date"] = current_time.strftime("%Y-%m-%d")

                # Add to historical data with current timestamp
                record = item.copy()
                record["timestamp"] = current_time.isoformat()

                # Convert srch_rqst to integer if possible
                if "srch_rqst" in record:
                    try:
                        record["srch_rqst"] = int(record["srch_rqst"])
                    except (ValueError, TypeError):
                        pass

                self.historical_data[f"trends_{data_type}"].append(record)

            logger.info(f"Processed {len(data)} {data_type} trend records")
        except Exception as e:
            logger.error(f"Error processing trends data: {e}")
            logger.error(traceback.format_exc())

    def _save_historical_data(self):
        """Save historical data to files"""
        try:
            for data_key, data_value in self.historical_data.items():
                file_path = os.path.join(
                    "data/historical", f"{data_key}_historical.json"
                )
                with open(file_path, "w") as f:
                    json.dump(data_value, f)

            logger.info("Historical data saved to files")
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")
            logger.error(traceback.format_exc())

    def load_historical_data(self):
        """Load historical data from files"""
        try:
            self.historical_data = {}

            for data_type in ["ward", "ca"]:
                file_path = os.path.join(
                    "data/historical", f"trends_{data_type}_historical.json"
                )
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        self.historical_data[f"trends_{data_type}"] = json.load(f)
                    logger.info(
                        f"Loaded {len(self.historical_data[f'trends_{data_type}'])} {data_type} trend records"
                    )
                else:
                    self.historical_data[f"trends_{data_type}"] = []
                    logger.info(f"No historical {data_type} trend records found")

            # If we don't have enough historical data, generate some synthetic data
            self._ensure_minimum_data()

            return True
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            logger.error(traceback.format_exc())
            return False

    def _ensure_minimum_data(self):
        """Ensure we have at least some data by generating synthetic data if needed"""
        min_records = 48  # At least 48 records (2 days of hourly data)

        for data_type in ["ward", "ca"]:
            key = f"trends_{data_type}"
            if (
                key not in self.historical_data
                or len(self.historical_data[key]) < min_records
            ):
                logger.info(
                    f"Not enough {data_type} historical data, generating synthetic data"
                )
                self._generate_synthetic_data(data_type)

    def _generate_synthetic_data(self, data_type):
        """Generate synthetic data for training"""
        key = f"trends_{data_type}"

        # Start from 3 days ago and generate hourly data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=3)

        synthetic_data = []

        # Generate for ward 1-5 or CA 1-3
        regions = range(1, 6) if data_type == "ward" else range(1, 4)

        for region_id in regions:
            region_name = str(region_id)

            # Create time range
            current_time = start_time
            while current_time < end_time:
                hour = current_time.hour
                day_of_week = current_time.weekday()

                # Higher at rush hours
                hour_factor = 1.0
                if 7 <= hour <= 10:  # Morning rush
                    hour_factor = 2.0
                elif 16 <= hour <= 19:  # Evening rush
                    hour_factor = 2.5
                elif 0 <= hour <= 5:  # Late night / early morning
                    hour_factor = 0.3

                # Weekend vs. weekday
                day_factor = 0.8 if day_of_week >= 5 else 1.2

                # Base demand varies by region
                base_demand = 10 + region_id * 5

                # Calculate request count with some randomness
                search_requests = int(
                    base_demand
                    * hour_factor
                    * day_factor
                    * (0.8 + 0.4 * np.random.random())
                )

                # Create record
                record = {
                    f"{data_type}_num": region_name,
                    "date": current_time.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "srch_rqst": search_requests,
                    "timestamp": current_time.isoformat(),
                }

                synthetic_data.append(record)

                # Move to next hour
                current_time += timedelta(hours=1)

        # Update historical data
        if key in self.historical_data:
            self.historical_data[key].extend(synthetic_data)
        else:
            self.historical_data[key] = synthetic_data

        logger.info(
            f"Generated {len(synthetic_data)} synthetic records for {data_type}"
        )

    def get_historical_trends_data(self, data_type="ward"):
        """Get historical trends data as a DataFrame"""
        try:
            key = f"trends_{data_type}"
            if key not in self.historical_data:
                logger.warning(f"No historical trends data available for {data_type}")
                return pd.DataFrame()

            df = pd.DataFrame(self.historical_data[key])
            if df.empty:
                return df

            # Rename columns for consistency
            region_col = "ward_num" if data_type == "ward" else "ac_num"
            if f"{data_type}_num" in df.columns and region_col not in df.columns:
                df[region_col] = df[f"{data_type}_num"]

            # Ensure numeric types
            if "srch_rqst" in df.columns:
                df["srch_rqst"] = (
                    pd.to_numeric(df["srch_rqst"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )

            if "hour" in df.columns:
                df["hour"] = (
                    pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
                )

            # Convert date and hour to datetime
            if "date" in df.columns and "hour" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["datetime"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")

            # Convert timestamp to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Sort by datetime
            if "datetime" in df.columns:
                df = df.sort_values("datetime")

            return df
        except Exception as e:
            logger.error(f"Error getting historical trends data: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()


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
            # Get historical data as DataFrame
            df = self.data_manager.get_historical_trends_data(data_type)

            if df.empty:
                logger.warning("No historical data available for modeling")
                return None

            # Check if we have the search request column
            if "srch_rqst" not in df.columns:
                logger.warning("No search request data available in historical data")
                return None

            # Determine region column based on data type
            region_col = "ward_num" if data_type == "ward" else "ac_num"

            if region_col not in df.columns:
                logger.warning(f"No {region_col} column in historical data")
                return None

            # Create time features
            df["hour_of_day"] = df["datetime"].dt.hour
            df["day_of_week"] = df["datetime"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
            df["is_morning_rush"] = df["hour_of_day"].apply(
                lambda x: 1 if 7 <= x <= 10 else 0
            )
            df["is_evening_rush"] = df["hour_of_day"].apply(
                lambda x: 1 if 16 <= x <= 19 else 0
            )

            # Group by region and datetime to get total search requests
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

            # Convert region to string
            agg_df[region_col] = agg_df[region_col].astype(str)

            # Create sorted time index and lag features
            agg_df = agg_df.sort_values([region_col, "datetime"])

            # Create lag features for each region separately
            regions = agg_df[region_col].unique()

            # Initialize lag and rolling mean columns
            for lag in range(1, 25):  # Lags from 1 to 24 hours
                agg_df[f"lag_{lag}"] = 0

            for window in [3, 6, 12, 24]:
                agg_df[f"rolling_mean_{window}h"] = 0

            # Process each region
            for region in regions:
                # Get data for this region
                mask = agg_df[region_col] == region
                region_df = agg_df[mask].copy()

                if len(region_df) <= 1:
                    continue

                # Create lag features
                for lag in range(1, 25):  # Lags from 1 to 24 hours
                    if lag < len(region_df):
                        agg_df.loc[mask, f"lag_{lag}"] = region_df["srch_rqst"].shift(
                            lag
                        )

                # Create rolling mean features
                for window in [3, 6, 12, 24]:
                    if len(region_df) >= window:
                        agg_df.loc[mask, f"rolling_mean_{window}h"] = (
                            region_df["srch_rqst"]
                            .rolling(window=window, min_periods=1)
                            .mean()
                        )

            # Fill NaN values
            agg_df = agg_df.fillna(0)

            logger.info(f"Prepared data for modeling with {len(agg_df)} records")
            return agg_df
        except Exception as e:
            logger.error(f"Error preparing data for modeling: {e}")
            logger.error(traceback.format_exc())
            return None

    def train_model(self, data_type="ward", force=False):
        """Train the forecasting model"""
        try:
            # Check if we need to retrain
            if (
                not force
                and self.model is not None
                and self.last_training_time is not None
            ):
                hours_since_training = (
                    datetime.now() - self.last_training_time
                ).total_seconds() / 3600
                if hours_since_training < 12:  # Only retrain every 12 hours
                    logger.info(
                        f"Skipping model training as it was trained {hours_since_training:.1f} hours ago"
                    )
                    return self.model

            # Prepare data for modeling
            df = self.prepare_data_for_modeling(data_type)

            if df is None or df.empty:
                logger.warning("No data available for model training")
                return None

            # Define features and target
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

            # Check if we have enough data points
            if len(df) < 2 * len(feature_cols):
                logger.warning(
                    f"Not enough data points ({len(df)}) compared to features ({len(feature_cols)})"
                )
                # Use a simpler model with fewer features
                feature_cols = [
                    col
                    for col in feature_cols
                    if col
                    in [
                        "hour_of_day",
                        "day_of_week",
                        "is_weekend",
                        "is_morning_rush",
                        "is_evening_rush",
                        "lag_1",
                        "lag_2",
                        "rolling_mean_3h",
                    ]
                ]

            self.features = feature_cols

            # Drop rows with missing values
            modeling_df = df.dropna(subset=feature_cols + ["srch_rqst"])

            if len(modeling_df) < 10:
                logger.warning(
                    f"Too few records ({len(modeling_df)}) for training after removing NaNs"
                )
                return None

            X = modeling_df[feature_cols]
            y = modeling_df["srch_rqst"]

            # Train a Random Forest model with appropriate settings
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=min(10, len(feature_cols)),
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,  # Use all available cores
            )
            self.model.fit(X, y)

            # Save the trained model
            model_path = os.path.join("data/models", f"rf_model_{data_type}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)

            # Save the feature list
            features_path = os.path.join("data/models", f"features_{data_type}.json")
            with open(features_path, "w") as f:
                json.dump(self.features, f)

            self.last_training_time = datetime.now()

            # Calculate feature importance
            feature_importance = pd.DataFrame(
                {"Feature": X.columns, "Importance": self.model.feature_importances_}
            ).sort_values("Importance", ascending=False)

            logger.info(
                f"Model trained successfully with {len(X)} data points and {len(feature_cols)} features"
            )
            detailed_logger.info(
                f"\nTop 10 most important features:\n{feature_importance.head(10)}"
            )

            return self.model
        except Exception as e:
            logger.error(f"Error training model: {e}")
            logger.error(traceback.format_exc())
            return None

    def load_model(self, data_type="ward"):
        """Load a previously trained model"""
        try:
            model_path = os.path.join("data/models", f"rf_model_{data_type}.pkl")
            features_path = os.path.join("data/models", f"features_{data_type}.json")

            if os.path.exists(model_path) and os.path.exists(features_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)

                with open(features_path, "r") as f:
                    self.features = json.load(f)

                logger.info(f"Loaded model from {model_path}")
                return True
            else:
                logger.warning(f"No model file found at {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            return False

    def forecast_next_hours(self, hours=24, data_type="ward", region=None):
        """Forecast ride requests for the next specified hours"""
        try:
            # Make sure we have a model
            if self.model is None:
                model_loaded = self.load_model(data_type)
                if not model_loaded:
                    logger.info("No pre-trained model found, training a new one")
                    self.train_model(data_type)

                if self.model is None:
                    logger.error("Failed to load or train a model")
                    return None

            # Get historical data to use as a basis for forecasting
            df = self.prepare_data_for_modeling(data_type)

            if df is None or df.empty:
                logger.warning("No historical data available for forecasting")
                return None

            # Define region column based on data type
            region_col = "ward_num" if data_type == "ward" else "ac_num"

            # Filter to a specific region if specified
            if region is not None:
                # Convert region to string if it's not already
                region_str = str(region)
                df = df[df[region_col] == region_str]

                if df.empty:
                    logger.warning(f"No data available for region {region}")
                    return None

            # Get all regions or just the specified one
            if region is None:
                regions = df[region_col].unique()
            else:
                # Convert region to string if it's not already
                region_str = str(region)
                regions = [region_str]

            forecasts = []

            for curr_region in regions:
                # Filter data for this region
                region_df = df[df[region_col] == curr_region].copy()
                if len(region_df) == 0:
                    continue

                # Get the last timestamp
                last_datetime = region_df["datetime"].max()

                # Create time points for forecasting
                forecast_times = pd.date_range(
                    start=last_datetime + timedelta(hours=1), periods=hours, freq="H"
                )

                # Create a DataFrame for future time points
                future_df = pd.DataFrame(index=range(hours))
                future_df["datetime"] = forecast_times
                future_df[region_col] = curr_region
                future_df["hour_of_day"] = future_df["datetime"].dt.hour
                future_df["day_of_week"] = future_df["datetime"].dt.dayofweek
                future_df["is_weekend"] = future_df["day_of_week"].apply(
                    lambda x: 1 if x >= 5 else 0
                )
                future_df["is_morning_rush"] = future_df["hour_of_day"].apply(
                    lambda x: 1 if 7 <= x <= 10 else 0
                )
                future_df["is_evening_rush"] = future_df["hour_of_day"].apply(
                    lambda x: 1 if 16 <= x <= 19 else 0
                )

                # Get the most recent search request values
                recent_values = (
                    region_df.sort_values("datetime").tail(24)["srch_rqst"].values
                )

                # Initialize features
                for feature in self.features:
                    # For lag features, initialize with recent values if available
                    if feature.startswith("lag_"):
                        lag = int(feature.split("_")[1])
                        if lag <= len(recent_values):
                            future_df[feature] = 0  # Will be updated in the loop
                        else:
                            future_df[feature] = 0
                    # For rolling mean features
                    elif feature.startswith("rolling_mean_"):
                        window = int(feature.split("_")[2].replace("h", ""))
                        if len(recent_values) > 0:
                            window_size = min(window, len(recent_values))
                            future_df[feature] = np.mean(recent_values[-window_size:])
                        else:
                            future_df[feature] = 0
                    # For other features, they're already initialized above

                # Make predictions iteratively
                predictions = []

                for i in range(hours):
                    # Update lag features based on previous predictions
                    if i > 0:
                        # Update lag features
                        for lag in range(1, min(i + 1, 24) + 1):
                            lag_col = f"lag_{lag}"
                            if lag_col in self.features:
                                future_df.loc[i, lag_col] = predictions[i - lag]

                        # Update rolling mean features
                        for window in [3, 6, 12, 24]:
                            rolling_col = f"rolling_mean_{window}h"
                            if rolling_col in self.features:
                                # Get values for the window
                                values = []
                                for w in range(min(window, i + 1)):
                                    values.append(predictions[i - w - 1])

                                if values:
                                    future_df.loc[i, rolling_col] = np.mean(values)

                    # Prepare the feature vector
                    X_pred = future_df.iloc[i : i + 1][self.features]

                    # Make prediction
                    try:
                        prediction = self.model.predict(X_pred)[0]
                        # Ensure non-negative integer
                        prediction = max(0, int(round(prediction)))
                    except Exception as e:
                        logger.error(f"Error making prediction: {e}")
                        prediction = 0

                    predictions.append(prediction)

                # Create result DataFrame for this region
                result = pd.DataFrame(
                    {
                        "datetime": forecast_times,
                        region_col: curr_region,
                        "forecast_requests": predictions,
                    }
                )

                forecasts.append(result)

            # Combine forecasts for all regions
            if forecasts:
                combined_forecast = pd.concat(forecasts, ignore_index=True)

                # Save the forecast
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                region_suffix = f"_{region}" if region is not None else ""
                forecast_path = os.path.join(
                    "data/forecasts",
                    f"forecast_{data_type}{region_suffix}_{timestamp}.csv",
                )
                combined_forecast.to_csv(forecast_path, index=False)

                # Add to forecast history
                self.forecast_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "data_type": data_type,
                        "region": region,
                        "hours": hours,
                        "path": forecast_path,
                    }
                )

                # Save forecast history
                with open(
                    os.path.join("data/forecasts", "forecast_history.json"), "w"
                ) as f:
                    json.dump(self.forecast_history, f)

                logger.info(
                    f"Generated forecast for {len(regions)} regions over the next {hours} hours"
                )
                return combined_forecast
            else:
                logger.warning("No forecasts generated")
                return None
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            logger.error(traceback.format_exc())
            return None

    def plot_forecast(
        self, forecast_df, data_type="ward", region=None, show_historical=True
    ):
        """Plot the historical data and the forecast"""
        try:
            if forecast_df is None or forecast_df.empty:
                logger.warning("No forecast data available for plotting")
                return False

            # Create plot
            plt.figure(figsize=(12, 6))

            # Determine region column based on data type
            region_col = "ward_num" if data_type == "ward" else "ac_num"

            # If no specific region is specified, use the first one in the forecast
            if region is None and region_col in forecast_df.columns:
                regions = forecast_df[region_col].unique()
                if len(regions) > 0:
                    region = regions[0]

            # Filter to the specified region
            if region is not None and region_col in forecast_df.columns:
                forecast_df = forecast_df[forecast_df[region_col] == str(region)].copy()

            # Get historical data if requested
            if show_historical:
                historical_df = self.data_manager.get_historical_trends_data(data_type)

                if not historical_df.empty and region_col in historical_df.columns:
                    # Filter to the specified region
                    if region is not None:
                        region_str = str(region)
                        historical_df = historical_df[
                            historical_df[region_col] == region_str
                        ]

                    # Aggregate by datetime if needed
                    if len(historical_df) > 0:
                        # Get only recent history (last 48 hours)
                        min_datetime = forecast_df["datetime"].min() - timedelta(
                            hours=48
                        )
                        recent_history = historical_df[
                            historical_df["datetime"] >= min_datetime
                        ]

                        # Aggregate by datetime
                        agg_historical = (
                            recent_history.groupby("datetime")["srch_rqst"]
                            .sum()
                            .reset_index()
                        )

                        # Plot historical data
                        plt.plot(
                            agg_historical["datetime"],
                            agg_historical["srch_rqst"],
                            label="Historical",
                            color="blue",
                            alpha=0.7,
                        )

            # Plot forecast
            if (
                "datetime" in forecast_df.columns
                and "forecast_requests" in forecast_df.columns
            ):
                plt.plot(
                    forecast_df["datetime"],
                    forecast_df["forecast_requests"],
                    label="Forecast",
                    color="red",
                    linestyle="--",
                    marker="o",
                    markersize=4,
                )

            # Add shading for confidence interval (simple approach)
            if (
                "datetime" in forecast_df.columns
                and "forecast_requests" in forecast_df.columns
            ):
                upper_bound = forecast_df["forecast_requests"] * 1.2  # 20% higher
                lower_bound = forecast_df["forecast_requests"] * 0.8  # 20% lower
                plt.fill_between(
                    forecast_df["datetime"],
                    lower_bound,
                    upper_bound,
                    color="red",
                    alpha=0.2,
                )

            # Add title and labels
            region_text = (
                f" for {region_col.replace('_', ' ').title()} {region}"
                if region is not None
                else ""
            )
            plt.title(f"Ride Request Forecast{region_text}")
            plt.xlabel("Time")
            plt.ylabel("Number of Search Requests")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Format x-axis to show dates nicely
            plt.gcf().autofmt_xdate()

            plt.tight_layout()

            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            region_suffix = f"_{region}" if region is not None else ""
            plot_path = os.path.join(
                "data/visualizations",
                f"forecast_{data_type}{region_suffix}_{timestamp}.png",
            )
            plt.savefig(plot_path)

            # Also save a "latest" version for easy access
            latest_path = os.path.join(
                "data/visualizations", f"forecast_{data_type}{region_suffix}_latest.png"
            )
            plt.savefig(latest_path)

            plt.close()

            logger.info(f"Forecast plot saved to {plot_path}")
            return True
        except Exception as e:
            logger.error(f"Error plotting forecast: {e}")
            logger.error(traceback.format_exc())
            return False


class ForecastService:
    """Main service to run the forecasting process"""

    def __init__(self):
        self.data_manager = RideDataManager()
        self.forecaster = RideRequestForecast(self.data_manager)
        self.last_fetch_time = None
        self.last_forecast_time = None

    def initialize(self):
        """Initialize the service"""
        try:
            logger.info("Initializing forecast service")

            # Create necessary directories
            os.makedirs("data", exist_ok=True)
            os.makedirs("data/historical", exist_ok=True)
            os.makedirs("data/models", exist_ok=True)
            os.makedirs("data/forecasts", exist_ok=True)
            os.makedirs("data/visualizations", exist_ok=True)

            # Load historical data
            self.data_manager.load_historical_data()

            # Load pre-trained model if available
            model_loaded = False
            for data_type in ["ward", "ca"]:
                if self.forecaster.load_model(data_type):
                    model_loaded = True

            # Fetch initial data
            self.fetch_data()

            # Train model if not loaded
            if not model_loaded:
                logger.info("No pre-trained models found, training new models")
                self.forecaster.train_model("ward")
                self.forecaster.train_model("ca")

            logger.info("Forecast service initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing forecast service: {e}")
            logger.error(traceback.format_exc())
            return False

    def fetch_data(self):
        """Fetch the latest data"""
        try:
            logger.info("Fetching latest data")
            fetched_data = self.data_manager.fetch_all_endpoints()
            self.last_fetch_time = datetime.now()

            # Check if we need to retrain the model
            hours_since_training = float("inf")
            if self.forecaster.last_training_time:
                hours_since_training = (
                    datetime.now() - self.forecaster.last_training_time
                ).total_seconds() / 3600

            if hours_since_training > 12:  # Retrain every 12 hours
                logger.info("Retraining model with new data")
                self.forecaster.train_model("ward")
                self.forecaster.train_model("ca")

            logger.info("Data fetch completed")
            return True
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            logger.error(traceback.format_exc())
            return False

    def generate_forecast(self, hours=24):
        """Generate a new forecast"""
        try:
            logger.info(f"Generating forecast for the next {hours} hours")

            # Generate ward-level forecast
            ward_forecast = self.forecaster.forecast_next_hours(hours, "ward")
            if ward_forecast is not None:
                self.forecaster.plot_forecast(ward_forecast, "ward")

                # Create individual plots for top wards
                top_wards = (
                    ward_forecast.groupby("ward_num")["forecast_requests"]
                    .sum()
                    .nlargest(5)
                    .index.tolist()
                )
                for ward in top_wards:
                    self.forecaster.plot_forecast(ward_forecast, "ward", region=ward)

            # Generate CA-level forecast
            ca_forecast = self.forecaster.forecast_next_hours(hours, "ca")
            if ca_forecast is not None:
                self.forecaster.plot_forecast(ca_forecast, "ca")

                # Create individual plots for top CAs
                top_cas = (
                    ca_forecast.groupby("ac_num")["forecast_requests"]
                    .sum()
                    .nlargest(3)
                    .index.tolist()
                )
                for ca in top_cas:
                    self.forecaster.plot_forecast(ca_forecast, "ca", region=ca)

            self.last_forecast_time = datetime.now()

            # Generate dashboard
            self.generate_dashboard(ward_forecast, ca_forecast)

            logger.info("Forecast generation completed")
            return True
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            logger.error(traceback.format_exc())
            return False

    def generate_dashboard(self, ward_forecast=None, ca_forecast=None):
        """Generate an HTML dashboard with forecasts"""
        try:
            logger.info("Generating dashboard")

            # Create a simple dashboard file
            dashboard_path = os.path.join("data/visualizations", "dashboard.html")
            with open(dashboard_path, "w") as f:
                f.write(
                    "<html><body><h1>Forecast Dashboard</h1><p>Generated at: "
                    + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    + "</p></body></html>"
                )

            logger.info(f"Dashboard generated at {dashboard_path}")
            return dashboard_path
        except Exception as e:
            logger.error(f"Error generating dashboard: {e}")
            logger.error(traceback.format_exc())
            return None

    def scheduled_fetch(self):
        """Function to be called by the scheduler for fetching data"""
        try:
            logger.info("Running scheduled data fetch")
            success = self.fetch_data()
            if success:
                logger.info("Scheduled fetch completed successfully")
            else:
                logger.error("Scheduled fetch failed")
        except Exception as e:
            logger.error(f"Error in scheduled_fetch: {e}")
            logger.error(traceback.format_exc())

    def scheduled_forecast(self):
        """Function to be called by the scheduler for generating forecasts"""
        try:
            logger.info("Running scheduled forecast generation")
            success = self.generate_forecast()
            if success:
                logger.info("Scheduled forecast completed successfully")
            else:
                logger.error("Scheduled forecast failed")
        except Exception as e:
            logger.error(f"Error in scheduled_forecast: {e}")
            logger.error(traceback.format_exc())

    def run(self):
        """Run the service with scheduled tasks"""
        try:
            logger.info("Starting forecast service")

            # Initialize the service
            if not self.initialize():
                logger.error("Failed to initialize forecast service")
                return False

            # Generate an initial forecast
            self.generate_forecast()

            # Schedule data fetching every 15 minutes
            schedule.every(15).minutes.do(self.scheduled_fetch)

            # Schedule forecast generation every hour
            schedule.every(1).hours.do(self.scheduled_forecast)

            logger.info("Forecast service started successfully")
            logger.info("- Data will be fetched every 15 minutes")
            logger.info("- Forecasts will be generated hourly")

            # Run the scheduling loop
            while True:
                schedule.run_pending()
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error running forecast service: {e}")
            logger.error(traceback.format_exc())
            return False


def main():
    """Main function to run the forecast service"""
    try:
        # Create and run the service
        service = ForecastService()

        # Initialize and run the service
        service.initialize()
        service.generate_forecast()
        service.run()

        return 0
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
