import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import holidays
import logging
import pickle
from datetime import datetime, timedelta

# Get the detailed logger
try:
    detailed_logger = logging.getLogger("train_detailed")
except:
    # Fallback if the logger doesn't exist
    detailed_logger = logging.getLogger("model")
    detailed_logger.setLevel(logging.INFO)
    if not detailed_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        detailed_logger.addHandler(handler)


def load_and_preprocess_data(trends_file, driver_file, funnel_file=None):
    """
    Load and preprocess data from the given files

    Args:
        trends_file: Path to the trends data file
        driver_file: Path to the driver data file
        funnel_file: Path to the funnel data file (optional)

    Returns:
        Preprocessed DataFrame with all features
    """
    try:
        trends_df = pd.read_json(trends_file)
        driver_df = pd.read_json(driver_file)

        detailed_logger.info("\nData Loading Diagnostics:")
        detailed_logger.info(f"Trends DataFrame columns: {trends_df.columns}")
        detailed_logger.info(f"Driver DataFrame columns: {driver_df.columns}")
        detailed_logger.info("\nTrends DataFrame sample:")
        detailed_logger.info(f"{trends_df.head()}")
        detailed_logger.info("\nDriver DataFrame sample:")
        detailed_logger.info(f"{driver_df.head()}")

        if funnel_file:
            funnel_df = pd.read_json(funnel_file)
            detailed_logger.info("\nFunnel DataFrame columns: {funnel_df.columns}")
            detailed_logger.info("\nFunnel DataFrame sample:")
            detailed_logger.info(f"{funnel_df.head()}")

            # Check earnings data in funnel_df
            detailed_logger.info("\nEarnings data in funnel_df:")
            detailed_logger.info(
                f"Min: {funnel_df['earning'].min()}, Max: {funnel_df['earning'].max()}, Mean: {funnel_df['earning'].mean()}"
            )
            detailed_logger.info(
                f"Sample earnings values: {funnel_df['earning'].sample(10).tolist()}"
            )
            detailed_logger.info(
                f"Number of zero earnings: {(funnel_df['earning'] == 0).sum()} out of {len(funnel_df)}"
            )

            # Check done_ride data in funnel_df
            detailed_logger.info("\nDone ride data in funnel_df:")
            detailed_logger.info(
                f"Min: {funnel_df['done_ride'].min()}, Max: {funnel_df['done_ride'].max()}, Mean: {funnel_df['done_ride'].mean()}"
            )
            detailed_logger.info(
                f"Sample done_ride values: {funnel_df['done_ride'].sample(10).tolist()}"
            )
            detailed_logger.info(
                f"Number of zero done_rides: {(funnel_df['done_ride'] == 0).sum()} out of {len(funnel_df)}"
            )

            # Check vehicle types
            detailed_logger.info(
                f"\nUnique vehicle types in funnel data: {funnel_df['vehicle_type'].unique()}"
            )
        else:
            funnel_df = None

        # Merge the dataframes
        if funnel_df is not None:
            # First merge trends and driver data
            merged_df = pd.merge(
                trends_df,
                driver_df,
                on=(
                    ["ward_num", "vehicle_type"]
                    if "vehicle_type" in trends_df.columns
                    else "ward_num"
                ),
                how="inner",
            )

            # Then merge with funnel data
            merged_df = pd.merge(
                merged_df,
                funnel_df,
                on=["ward_num", "vehicle_type"],
                how="inner",
            )
        else:
            # Just merge trends and driver data
            merged_df = pd.merge(
                trends_df,
                driver_df,
                on=(
                    ["ward_num", "vehicle_type"]
                    if "vehicle_type" in trends_df.columns
                    else "ward_num"
                ),
                how="inner",
            )

        detailed_logger.info("\nMerged DataFrame sample:")
        detailed_logger.info(f"{merged_df.head()}")

        # Check for NaN values
        detailed_logger.info("\nColumns with NaN values:")
        detailed_logger.info(f"{merged_df.isna().sum()}")

        # Calculate avg_earning_per_ride
        if "earning" in merged_df.columns and "done_ride" in merged_df.columns:
            # Avoid division by zero
            merged_df["avg_earning_per_ride"] = merged_df.apply(
                lambda row: (
                    row["earning"] / row["done_ride"] if row["done_ride"] > 0 else 0
                ),
                axis=1,
            )

            # Log the calculated values
            detailed_logger.info("\nEarnings data in funnel_df:")
            detailed_logger.info(
                f"Min: {merged_df['earning'].min()}, Max: {merged_df['earning'].max()}, Mean: {merged_df['earning'].mean()}"
            )
            detailed_logger.info(
                f"Sample earnings values: {merged_df['earning'].head(10).tolist()}"
            )
            detailed_logger.info(
                f"Number of zero earnings: {(merged_df['earning'] == 0).sum()} out of {len(merged_df)}"
            )

            detailed_logger.info("\nDone ride data in funnel_df:")
            detailed_logger.info(
                f"Min: {merged_df['done_ride'].min()}, Max: {merged_df['done_ride'].max()}, Mean: {merged_df['done_ride'].mean()}"
            )
            detailed_logger.info(
                f"Sample done_ride values: {merged_df['done_ride'].head(10).tolist()}"
            )
            detailed_logger.info(
                f"Number of zero done_rides: {(merged_df['done_ride'] == 0).sum()} out of {len(merged_df)}"
            )

            detailed_logger.info("\nCalculated avg_earning_per_ride:")
            detailed_logger.info(
                f"Min: {merged_df['avg_earning_per_ride'].min()}, Max: {merged_df['avg_earning_per_ride'].max()}, Mean: {merged_df['avg_earning_per_ride'].mean()}"
            )
            detailed_logger.info(
                f"Sample avg_earning_per_ride values: {merged_df['avg_earning_per_ride'].head(10).tolist()}"
            )
            detailed_logger.info(
                f"Number of zero avg_earning_per_ride: {(merged_df['avg_earning_per_ride'] == 0).sum()} out of {len(merged_df)}"
            )

        # Calculate conversion and completion rates
        if all(
            col in merged_df.columns
            for col in ["srch_rqst", "booking", "done_ride", "cancel_ride"]
        ):
            # Avoid division by zero
            merged_df["conversion_rate"] = merged_df.apply(
                lambda row: (
                    row["booking"] / row["srch_rqst"] if row["srch_rqst"] > 0 else 0
                ),
                axis=1,
            )

            merged_df["completion_rate"] = merged_df.apply(
                lambda row: (
                    row["done_ride"] / row["booking"] if row["booking"] > 0 else 0
                ),
                axis=1,
            )

            merged_df["cancellation_rate"] = merged_df.apply(
                lambda row: (
                    row["cancel_ride"] / row["booking"] if row["booking"] > 0 else 0
                ),
                axis=1,
            )

            # Calculate rider satisfaction proxy (ratio of rider cancellations to total rides)
            if "rider_cancel" in merged_df.columns:
                merged_df["rider_satisfaction_proxy"] = merged_df.apply(
                    lambda row: (
                        1 - (row["rider_cancel"] / row["booking"])
                        if row["booking"] > 0
                        else 0
                    ),
                    axis=1,
                )

        # Log avg_fare data if available
        if "avg_fare" in merged_df.columns:
            detailed_logger.info("\nAvg fare data in funnel_df:")
            detailed_logger.info(
                f"Min: {merged_df['avg_fare'].min()}, Max: {merged_df['avg_fare'].max()}, Mean: {merged_df['avg_fare'].mean()}"
            )
            detailed_logger.info(
                f"Sample avg_fare values: {merged_df['avg_fare'].head(10).tolist()}"
            )
            detailed_logger.info(
                f"Number of zero avg_fare: {(merged_df['avg_fare'] == 0).sum()} out of {len(merged_df)}"
            )

        # Log the final columns
        detailed_logger.info("\nFinal Merged DataFrame columns:")
        detailed_logger.info(f"{merged_df.columns}")

        # Log a sample of key metrics
        key_metrics = [
            "ward_num",
            "vehicle_type",
            "active_drvr",
            "srch_rqst",
            "done_ride",
            "conversion_rate",
            "rider_satisfaction_proxy",
        ]
        available_metrics = [col for col in key_metrics if col in merged_df.columns]

        if available_metrics:
            detailed_logger.info("\nSample of key metrics:")
            detailed_logger.info(f"{merged_df[available_metrics].head()}")

        return merged_df

    except Exception as e:
        detailed_logger.error(f"Error in load_and_preprocess_data: {e}")
        import traceback

        detailed_logger.error(traceback.format_exc())
        return None


def create_features(df):
    """
    Create features for model training

    Args:
        df: DataFrame with preprocessed data

    Returns:
        DataFrame with features for model training
    """
    try:
        df = df.copy()

        # Convert string boolean values to actual boolean values
        for col in df.columns:
            if df[col].dtype == "object":
                # Check if column contains only 'TRUE' and 'FALSE' values
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 2 and all(
                    val in ["TRUE", "FALSE"] for val in unique_vals
                ):
                    df[col] = df[col].map({"TRUE": True, "FALSE": False})
                    detailed_logger.info(
                        f"Converted string boolean values in column {col} to actual boolean values"
                    )

        # Create a dictionary to store new features
        features = {}

        # Convert date to datetime if it's not already
        if "date" in df.columns:
            features["date"] = pd.to_datetime(df["date"])

            # Extract date components
            features["day_of_week"] = features["date"].dt.dayofweek
            features["day_of_month"] = features["date"].dt.day
            features["month"] = features["date"].dt.month
            features["is_weekend"] = features["day_of_week"].apply(
                lambda x: 1 if x >= 5 else 0
            )

            # Add holiday feature
            india_holidays = holidays.India()
            features["is_holiday"] = features["date"].apply(
                lambda date: 1 if date in india_holidays else 0
            )

        # Hour features
        if "hour" in df.columns:
            features["hour"] = df["hour"].astype(int)

            # Time of day features
            features["morning_hour"] = ((df["hour"] >= 6) & (df["hour"] <= 10)).astype(
                int
            )
            features["evening_hour"] = ((df["hour"] >= 16) & (df["hour"] <= 20)).astype(
                int
            )
            features["night_hour"] = ((df["hour"] >= 21) | (df["hour"] <= 5)).astype(
                int
            )
            features["lunch_hour"] = ((df["hour"] >= 11) & (df["hour"] <= 15)).astype(
                int
            )

            # Peak hour feature
            features["is_peak_hour"] = (
                ((df["hour"] >= 7) & (df["hour"] <= 9))
                | ((df["hour"] >= 17) & (df["hour"] <= 19))
            ).astype(int)

            # Cyclical encoding of hour
            features["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            features["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Search request features
        if "srch_rqst" in df.columns:
            features["srch_rqst"] = df["srch_rqst"]
            features["srch_rqst_log"] = np.log1p(df["srch_rqst"])
            features["srch_rqst_scaled"] = df["srch_rqst"] / df["srch_rqst"].max()

            # Interaction features
            if "hour" in df.columns:
                features["srch_hour_interaction"] = df["srch_rqst"] * features["hour"]

            if "is_weekend" in features:
                features["srch_weekend_interaction"] = (
                    df["srch_rqst"] * features["is_weekend"]
                )

            if "is_peak_hour" in features:
                features["srch_peak_interaction"] = (
                    df["srch_rqst"] * features["is_peak_hour"]
                )

        # Driver features
        if "active_drvr" in df.columns:
            features["active_drvr"] = df["active_drvr"]

            # Supply-demand ratio
            if "srch_rqst" in df.columns:
                features["supply_demand_ratio"] = df["active_drvr"] / (
                    df["srch_rqst"] + 1e-6
                )
                features["demand_supply_ratio"] = df["srch_rqst"] / (
                    df["active_drvr"] + 1e-6
                )

        # Booking and conversion features
        if "booking" in df.columns and "srch_rqst" in df.columns:
            features["booking"] = df["booking"]
            features["booking_success_rate"] = df["booking"] / (df["srch_rqst"] + 1e-6)

        # Ride completion features
        if "done_ride" in df.columns and "booking" in df.columns:
            features["done_ride"] = df["done_ride"]
            features["completion_rate"] = df["done_ride"] / (df["booking"] + 1e-6)

        # Cancellation features
        if "cancel_ride" in df.columns and "booking" in df.columns:
            features["cancellation_rate"] = df["cancel_ride"] / (df["booking"] + 1e-6)

            if "drvr_cancel" in df.columns:
                features["driver_cancel_rate"] = df["drvr_cancel"] / (
                    df["cancel_ride"] + 1e-6
                )

            if "rider_cancel" in df.columns:
                features["rider_cancel_rate"] = df["rider_cancel"] / (
                    df["cancel_ride"] + 1e-6
                )

        # Earnings features
        if "earning" in df.columns and "done_ride" in df.columns:
            features["earning"] = df["earning"]
            features["avg_earning_per_ride"] = df["earning"] / (df["done_ride"] + 1e-6)

        # Distance features
        if "dist" in df.columns and "done_ride" in df.columns:
            features["avg_distance"] = df["dist"] / (df["done_ride"] + 1e-6)

        # Rider satisfaction
        if "rider_satisfaction_proxy" in df.columns:
            features["rider_satisfaction"] = df["rider_satisfaction_proxy"]
        elif "rider_cancel" in df.columns and "booking" in df.columns:
            features["rider_satisfaction"] = 1 - (
                df["rider_cancel"] / (df["booking"] + 1e-6)
            )

        # Queue effectiveness
        if "srch_fr_q" in df.columns and "srch_which_got_q" in df.columns:
            features["queue_effectiveness"] = df["srch_which_got_q"] / (
                df["srch_fr_q"] + 1e-6
            )

        # Vehicle type features
        if (
            "is_bike" in df.columns
            and "is_cab" in df.columns
            and "is_auto" in df.columns
        ):
            features["is_bike"] = df["is_bike"]
            features["is_cab"] = df["is_cab"]
            features["is_auto"] = df["is_auto"]

        # One-hot encode categorical features
        categorical_features = []
        if "ward_num" in df.columns:
            categorical_features.append("ward_num")
            features["ward_num"] = df["ward_num"]

        if "vehicle_type" in df.columns:
            categorical_features.append("vehicle_type")
            features["vehicle_type"] = df["vehicle_type"]

        # Create DataFrame from features dictionary
        feature_df = pd.DataFrame(features)

        # One-hot encode categorical features
        if categorical_features:
            for cat_feature in categorical_features:
                if cat_feature in feature_df.columns:
                    dummies = pd.get_dummies(
                        feature_df[cat_feature], prefix=cat_feature, drop_first=False
                    )
                    feature_df = pd.concat([feature_df, dummies], axis=1)

        # Add time-based features if date and hour are available
        if "date" in feature_df.columns and "hour" in feature_df.columns:
            # Sort by date and hour for time-based features
            df_sorted = df.sort_values(["date", "hour"])

            if "srch_rqst" in df_sorted.columns:
                # Lag features
                feature_df["srch_rqst_lag_1h"] = df_sorted["srch_rqst"].shift(1)
                feature_df["srch_rqst_lag_2h"] = df_sorted["srch_rqst"].shift(2)
                feature_df["srch_rqst_lag_1d"] = df_sorted["srch_rqst"].shift(24)

                # Rolling window features
                feature_df["srch_rqst_rolling_3h"] = (
                    df_sorted["srch_rqst"].rolling(window=3, min_periods=1).mean()
                )
                feature_df["srch_rqst_rolling_6h"] = (
                    df_sorted["srch_rqst"].rolling(window=6, min_periods=1).mean()
                )

                # Trend features
                feature_df["srch_hourly_change"] = df_sorted["srch_rqst"] - df_sorted[
                    "srch_rqst"
                ].shift(1)
                feature_df["srch_trend"] = np.sign(feature_df["srch_hourly_change"])

        # Fill NaN values
        for col in feature_df.columns:
            if feature_df[col].dtype == "object":
                feature_df[col] = feature_df[col].fillna("unknown")
            elif "lag" in col or "rolling" in col:
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())
            else:
                feature_df[col] = feature_df[col].fillna(0)

        return feature_df

    except Exception as e:
        detailed_logger.error(f"Error in create_features: {e}")
        import traceback

        detailed_logger.error(traceback.format_exc())
        return None


def train_models(df, target_variable="avg_earning_per_ride"):
    """
    Train multiple models on the given data

    Args:
        df: DataFrame with features
        target_variable: Target variable to predict

    Returns:
        Dictionary of trained models and their features
    """
    try:
        # Make a copy of the dataframe
        df = df.copy()

        # Check if target variable exists
        if target_variable not in df.columns:
            detailed_logger.error(
                f"Target variable '{target_variable}' not found in DataFrame"
            )
            detailed_logger.info(f"Available columns: {df.columns.tolist()}")
            raise ValueError(
                f"Target variable '{target_variable}' not found in DataFrame"
            )

        # Define columns to exclude from features
        exclude_cols = [
            "date",
            target_variable,
            "ward_num",
            "vehicle_type",
            "earning",
            "booking",
            "done_ride",
            "cancel_ride",
            "drvr_cancel",
            "rider_cancel",
            "srch_which_got_e",
            "srch_fr_q",
            "srch_which_got_q",
        ]

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Log feature columns
        detailed_logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

        # Split data into features and target
        X = df[feature_cols]
        y = df[target_variable]

        # Log data date range
        if "date" in df.columns:
            min_date = df["date"].min()
            max_date = df["date"].max()
            detailed_logger.info(f"Data date range: {min_date} to {max_date}")

        # Split data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        # Log data split sizes
        detailed_logger.info(f"X_train shape: {X_train.shape}")
        detailed_logger.info(f"y_train shape: {y_train.shape}")
        detailed_logger.info(f"X_val shape: {X_val.shape}")
        detailed_logger.info(f"y_val shape: {y_val.shape}")
        detailed_logger.info(f"X_test shape: {X_test.shape}")
        detailed_logger.info(f"y_test shape: {y_test.shape}")

        # Initialize models
        models = {
            "random_forest": RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features=0.7,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features=0.7,
                random_state=42,
            ),
        }

        # Train and evaluate models
        trained_models = {}
        for model_name, model in models.items():
            detailed_logger.info(f"\nTraining {model_name} model...")

            # Train model
            model.fit(X_train, y_train)

            # Evaluate on validation set
            y_pred_val = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            val_r2 = r2_score(y_val, y_pred_val)

            detailed_logger.info(f"{model_name} Validation RMSE: {val_rmse:.4f}")
            detailed_logger.info(f"{model_name} Validation R-squared: {val_r2:.4f}")

            # Evaluate on test set
            y_pred_test = model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_r2 = r2_score(y_test, y_pred_test)

            detailed_logger.info(f"{model_name} Test RMSE: {test_rmse:.4f}")
            detailed_logger.info(f"{model_name} Test R-squared: {test_r2:.4f}")

            # Get feature importances
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feature_importance = pd.DataFrame(
                    {"Feature": feature_cols, "Importance": importances}
                ).sort_values("Importance", ascending=False)

                detailed_logger.info(f"\n{model_name} Top 10 important features:")
                detailed_logger.info(f"{feature_importance.head(10)}")

            # Store trained model
            trained_models[model_name] = {
                "model": model,
                "features": feature_cols,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
            }

        # Determine best model based on validation RMSE
        best_model_name = min(
            trained_models, key=lambda k: trained_models[k]["val_rmse"]
        )
        detailed_logger.info(
            f"\nBest model: {best_model_name} with validation RMSE: {trained_models[best_model_name]['val_rmse']:.4f}"
        )

        # Add best model to the result
        trained_models["best_model"] = best_model_name

        return trained_models

    except Exception as e:
        detailed_logger.error(f"Error in train_models: {e}")
        import traceback

        detailed_logger.error(traceback.format_exc())
        return None


def predict_earnings(models, input_data, model_name=None):
    """
    Make predictions using the trained models

    Args:
        models: Dictionary of trained models from train_models
        input_data: DataFrame with input features
        model_name: Name of the model to use for prediction (if None, use the best model)

    Returns:
        DataFrame with predictions
    """
    try:
        # Make a copy of the input data
        input_data = input_data.copy()

        # Handle the case when models is not a dictionary
        if not isinstance(models, dict):
            detailed_logger.warning(
                "Models is not a dictionary, using the model directly"
            )
            # If models is a sklearn model directly, use it
            if hasattr(models, "predict"):
                model = models
                # Create dummy features list using all columns except obvious non-feature ones
                exclude_cols = [
                    "date",
                    "ward_num",
                    "vehicle_type",
                    "earning",
                    "dist",
                    "avg_dist_pr_trip",
                    "avg_fare",
                    "avg_earning_per_ride",
                    "predicted_earnings",
                ]
                features = [
                    col for col in input_data.columns if col not in exclude_cols
                ]

                # Convert any string boolean values to actual boolean values
                for col in input_data.columns:
                    if input_data[col].dtype == "object":
                        # Check if column contains only 'TRUE' and 'FALSE' values
                        unique_vals = input_data[col].dropna().unique()
                        if len(unique_vals) <= 2 and all(
                            val in ["TRUE", "FALSE"] for val in unique_vals
                        ):
                            input_data[col] = input_data[col].map(
                                {"TRUE": True, "FALSE": False}
                            )

                # Make predictions
                X = input_data[features]
                predictions = model.predict(X)
                input_data["predicted_earnings"] = predictions
                return input_data
            else:
                detailed_logger.error(
                    "Models is not a dictionary and does not have a predict method"
                )
                return None

        # If model_name is not specified, use the best model
        if model_name is None:
            if "best_model" in models:
                model_name = models["best_model"]
            else:
                # If best_model is not specified, use the first model
                model_name = list(models.keys())[0]

        # Check if the specified model exists
        if model_name not in models:
            detailed_logger.error(
                f"Model '{model_name}' not found in models dictionary"
            )
            raise ValueError(f"Model '{model_name}' not found in models dictionary")

        # Get the model and features
        model_info = models[model_name]
        model = model_info["model"]
        features = model_info["features"]

        # Check if all required features are in the input data
        missing_features = [f for f in features if f not in input_data.columns]
        if missing_features:
            detailed_logger.warning(
                f"Missing features in input data: {missing_features}"
            )
            detailed_logger.warning("These features will be filled with zeros")

            # Fill missing features with zeros
            for feature in missing_features:
                input_data[feature] = 0

        # Select only the features used by the model
        X = input_data[features]

        # Make predictions
        predictions = model.predict(X)

        # Add predictions to the input data
        input_data["predicted_earnings"] = predictions

        # Log prediction statistics
        detailed_logger.info(f"\nPrediction statistics using {model_name} model:")
        detailed_logger.info(
            f"Min: {predictions.min()}, Max: {predictions.max()}, Mean: {predictions.mean()}"
        )
        detailed_logger.info(f"Sample predictions: {predictions[:5]}")

        return input_data

    except Exception as e:
        detailed_logger.error(f"Error in predict_earnings: {e}")
        import traceback

        detailed_logger.error(traceback.format_exc())
        return None
