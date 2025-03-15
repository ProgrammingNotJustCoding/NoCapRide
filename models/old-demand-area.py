import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import holidays

def load_and_preprocess_data(trends_file, driver_file, funnel_file=None):
    try:
        trends_df = pd.read_json(trends_file)
        driver_df = pd.read_json(driver_file)

        print("\nData Loading Diagnostics:")
        print("Trends DataFrame columns:", trends_df.columns)
        print("Driver DataFrame columns:", driver_df.columns)
        print("\nTrends DataFrame sample:")
        print(trends_df.head())
        print("\nDriver DataFrame sample:")
        print(driver_df.head())

        if funnel_file:
            funnel_df = pd.read_json(funnel_file)
            print("\nFunnel DataFrame columns:", funnel_df.columns)
            print("\nFunnel DataFrame sample:")
            print(funnel_df.head())
            print(
                "\nUnique vehicle types in funnel data:",
                funnel_df["vehicle_type"].unique(),
            )

            merged_df = pd.merge(trends_df, driver_df, on=["ward_num"], how="left")

            valid_vehicle_types = driver_df["vehicle_type"].unique()
            funnel_df = funnel_df[funnel_df["vehicle_type"].isin(valid_vehicle_types)]

            merged_df = pd.merge(
                merged_df, funnel_df, on=["ward_num", "vehicle_type"], how="left"
            )

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

            print("\nMerged DataFrame sample:")
            print(merged_df.head())
            print("\nColumns with NaN values:")
            print(merged_df.isna().sum())

        else:
            merged_df = pd.merge(trends_df, driver_df, on=["ward_num"], how="left")

        driver_cols = ["active", "active_drvr", "drvr_notonride", "drvr_onride"]
        for col in driver_cols:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(0)

        if funnel_file:
            funnel_cols = [
                "srch_fr_e",
                "srch_which_got_e",
                "srch_fr_q",
                "srch_which_got_q",
                "booking",
                "ongng_ride",
                "done_ride",
                "rides",
                "cancel_ride",
                "drvr_cancel",
                "rider_cancel",
                "earning",
                "dist",
                "avg_dist_pr_trip",
                "avg_fare",
            ]
            for col in funnel_cols:
                if col in merged_df.columns:

                    if col in [
                        "booking",
                        "done_ride",
                        "cancel_ride",
                        "drvr_cancel",
                        "rider_cancel",
                    ]:
                        merged_df[col] = merged_df[col].fillna(0)

                    elif "rate" in col or "cnvr" in col:
                        merged_df[col] = merged_df[col].fillna(0.5)

                    elif col in ["earning", "avg_fare"]:
                        median_value = merged_df[col][merged_df[col] > 0].median()
                        merged_df[col] = merged_df[col].fillna(
                            median_value if not pd.isna(median_value) else 0
                        )
                    else:
                        merged_df[col] = merged_df[col].fillna(0)

        bool_cols = ["active", "is_bike", "is_cab", "is_auto"]
        for col in bool_cols:
            if col in merged_df.columns:
                merged_df[col] = (
                    merged_df[col].map({"TRUE": 1, "FALSE": 0}).fillna(0).astype(int)
                )

        merged_df["avg_earning_per_ride"] = merged_df["earning"] / (
            merged_df["done_ride"] + 1e-6
        )
        merged_df["conversion_rate"] = merged_df["booking"] / (
            merged_df["srch_rqst"] + 1e-6
        )
        merged_df["completion_rate"] = merged_df["done_ride"] / (
            merged_df["booking"] + 1e-6
        )
        merged_df["cancellation_rate"] = merged_df["cancel_ride"] / (
            merged_df["booking"] + 1e-6
        )

        merged_df["rider_satisfaction_proxy"] = 1 - (
            merged_df["rider_cancel"] / (merged_df["booking"] + 1e-6)
        )
        merged_df["rider_satisfaction_proxy"] = merged_df[
            "rider_satisfaction_proxy"
        ].clip(0, 1)

        print("\nFinal Merged DataFrame columns:", merged_df.columns)
        print("\nSample of key metrics:")
        print(
            merged_df[
                [
                    "ward_num",
                    "vehicle_type",
                    "earning",
                    "done_ride",
                    "avg_earning_per_ride",
                    "conversion_rate",
                    "rider_satisfaction_proxy",
                ]
            ].head()
        )
        return merged_df

    except Exception as e:
        print(f"Error in load_and_preprocess_data: {str(e)}")
        raise


def create_features(df):

    df = df.copy()

    new_features = {}

    new_features["date"] = pd.to_datetime(df["date"])
    new_features["hour"] = df["hour"].astype(int)

    new_features["day_of_week"] = new_features["date"].dt.dayofweek
    new_features["day_of_month"] = new_features["date"].dt.day
    new_features["month"] = new_features["date"].dt.month
    new_features["is_weekend"] = new_features["day_of_week"].apply(
        lambda x: 1 if x >= 5 else 0
    )

    new_features["morning_hour"] = ((df["hour"] >= 6) & (df["hour"] <= 10)).astype(int)
    new_features["evening_hour"] = ((df["hour"] >= 16) & (df["hour"] <= 20)).astype(int)
    new_features["night_hour"] = ((df["hour"] >= 21) | (df["hour"] <= 5)).astype(int)
    new_features["lunch_hour"] = ((df["hour"] >= 11) & (df["hour"] <= 15)).astype(int)

    new_features["hour_squared"] = df["hour"] ** 2

    new_features["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    new_features["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    if "srch_rqst" in df.columns:
        new_features["srch_rqst_log"] = np.log1p(df["srch_rqst"])
        new_features["srch_rqst_scaled"] = df["srch_rqst"] / df["srch_rqst"].max()
        new_features["srch_rqst_binned"] = pd.qcut(
            df["srch_rqst"], 5, labels=False, duplicates="drop"
        )

        new_features["srch_hour_interaction"] = df["srch_rqst"] * df["hour"]
        new_features["srch_weekend_interaction"] = (
            df["srch_rqst"] * new_features["is_weekend"]
        )

    india_holidays = holidays.India()
    new_features["is_holiday"] = new_features["date"].apply(
        lambda date: 1 if date in india_holidays else 0
    )

    if "booking" in df.columns and "srch_rqst" in df.columns:
        new_features["conversion_rate"] = df["booking"] / (df["srch_rqst"] + 1e-6)

    if "done_ride" in df.columns and "booking" in df.columns:
        new_features["completion_rate"] = df["done_ride"] / (df["booking"] + 1e-6)

    if "cancel_ride" in df.columns and "booking" in df.columns:
        new_features["cancellation_rate"] = df["cancel_ride"] / (df["booking"] + 1e-6)

        if "drvr_cancel" in df.columns and "rider_cancel" in df.columns:
            new_features["driver_cancel_rate"] = df["drvr_cancel"] / (
                df["cancel_ride"] + 1e-6
            )
            new_features["rider_cancel_rate"] = df["rider_cancel"] / (
                df["cancel_ride"] + 1e-6
            )

    if "earning" in df.columns and "done_ride" in df.columns:
        new_features["avg_earning_per_ride"] = df["earning"] / (df["done_ride"] + 1e-6)

    if "dist" in df.columns and "done_ride" in df.columns:
        new_features["avg_distance"] = df["dist"] / (df["done_ride"] + 1e-6)

    if "rider_cancel" in df.columns and "booking" in df.columns:
        new_features["rider_satisfaction_proxy"] = 1 - (
            df["rider_cancel"] / (df["booking"] + 1e-6)
        )

    if "active_drvr" in df.columns and "srch_rqst" in df.columns:
        new_features["supply_demand_ratio"] = df["active_drvr"] / (
            df["srch_rqst"] + 1e-6
        )

    if "srch_fr_q" in df.columns and "srch_which_got_q" in df.columns:
        new_features["queue_effectiveness"] = df["srch_which_got_q"] / (
            df["srch_fr_q"] + 1e-6
        )

    if "is_bike" in df.columns and "is_cab" in df.columns and "is_auto" in df.columns:
        conversion_rate = new_features.get(
            "conversion_rate", pd.Series(0, index=df.index)
        )
        new_features["bike_conversion"] = df["is_bike"] * conversion_rate
        new_features["cab_conversion"] = df["is_cab"] * conversion_rate
        new_features["auto_conversion"] = df["is_auto"] * conversion_rate

    categorical_features = ["ward_num", "vehicle_type"]
    categorical_dummies = pd.get_dummies(df[categorical_features], dummy_na=False)
    for col in categorical_dummies.columns:
        new_features[col] = categorical_dummies[col]

    df_sorted = df.sort_values(["date", "hour"])

    if "srch_rqst" in df.columns:
        new_features["srch_rqst_lag_1h"] = df_sorted["srch_rqst"].shift(1)
        new_features["srch_rqst_lag_2h"] = df_sorted["srch_rqst"].shift(2)
        new_features["srch_rqst_lag_1d"] = df_sorted["srch_rqst"].shift(24)

        new_features["srch_rqst_rolling_3h"] = (
            df_sorted["srch_rqst"].rolling(window=3, min_periods=1).mean()
        )
        new_features["srch_rqst_rolling_6h"] = (
            df_sorted["srch_rqst"].rolling(window=6, min_periods=1).mean()
        )

        new_features["srch_hourly_change"] = df_sorted["srch_rqst"] - df_sorted[
            "srch_rqst"
        ].shift(1)
        new_features["srch_trend"] = np.sign(new_features["srch_hourly_change"])

    for ward_col in [col for col in categorical_dummies.columns if "ward_num_" in col]:
        ward_id = ward_col.split("_")[-1]
        ward_mask = categorical_dummies[ward_col] == 1
        if ward_mask.any():
            lagged_values = pd.Series(index=df.index, dtype="float64")
            lagged_values.loc[ward_mask] = df_sorted.loc[ward_mask, "srch_rqst"].shift(
                1
            )
            new_features[f"srch_rqst_lag_1h_ward_{ward_id}"] = lagged_values

    if "srch_rqst" in df.columns and "active_drvr" in df.columns:
        new_features["demand_supply_ratio"] = df["srch_rqst"] / (
            df["active_drvr"] + 1e-6
        )

    if "booking" in df.columns and "srch_rqst" in df.columns:
        new_features["booking_success_rate"] = df["booking"] / (df["srch_rqst"] + 1e-6)

    new_features["is_peak_hour"] = df["hour"].apply(
        lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0
    )

    df_new = pd.DataFrame(new_features, index=df.index)

    for col in df_new.columns:
        if "ward_num_" in col or "vehicle_type_" in col:
            df_new[col] = df_new[col].fillna(0)
        elif "lag" in col or "rolling" in col:
            df_new[col] = df_new[col].fillna(df_new[col].median())
        else:
            df_new[col] = df_new[col].fillna(0)

    return df_new


def train_model(df, target_variable="demand_supply_ratio"):
    try:

        exclude_cols = [
            "date",
            target_variable,
            "booking",
            "done_ride",
            "earning",
            "cancel_ride",
            "drvr_cancel",
            "rider_cancel",
            "cnvr_rate",
            "bkng_cancel_rate",
            "q_accept_rate",
            "srch_which_got_e",
            "srch_fr_q",
            "srch_which_got_q",
        ]

        features = [col for col in df.columns if col not in exclude_cols]

        if target_variable not in df.columns:
            print(
                f"Warning: Target variable '{target_variable}' not found in DataFrame"
            )
            print("Available columns:", df.columns.tolist())
            raise KeyError(
                f"Target variable '{target_variable}' not found in DataFrame"
            )

        X = df[features]
        y = df[target_variable]

        min_date = df["date"].min()
        max_date = df["date"].max()
        print(f"Data date range: {min_date} to {max_date}")

        total_days = (max_date - min_date).days
        train_days = int(total_days * 0.7)
        val_days = int(total_days * 0.15)

        train_end_date = min_date + pd.Timedelta(days=train_days)
        val_end_date = train_end_date + pd.Timedelta(days=val_days)

        print(f"Train end date: {train_end_date}")
        print(f"Validation end date: {val_end_date}")

        X_train = X.loc[df["date"] < train_end_date]
        y_train = y.loc[df["date"] < train_end_date]
        X_val = X.loc[(df["date"] >= train_end_date) & (df["date"] < val_end_date)]
        y_val = y.loc[(df["date"] >= train_end_date) & (df["date"] < val_end_date)]
        X_test = X.loc[df["date"] >= val_end_date]
        y_test = y.loc[df["date"] >= val_end_date]

        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_val shape:", X_val.shape)
        print("y_val shape:", y_val.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

        if X_val.shape[0] == 0:
            print("Validation set is empty, readjusting splits...")
            train_size = int(len(X) * 0.7)
            val_size = int(len(X) * 0.15)

            all_indices = X.index.tolist()
            train_indices = all_indices[:train_size]
            val_indices = all_indices[train_size : train_size + val_size]
            test_indices = all_indices[train_size + val_size :]

            X_train, y_train = X.loc[train_indices], y.loc[train_indices]
            X_val, y_val = X.loc[val_indices], y.loc[val_indices]
            X_test, y_test = X.loc[test_indices], y.loc[test_indices]

            print("After adjustment:")
            print("X_train shape:", X_train.shape)
            print("X_val shape:", X_val.shape)
            print("X_test shape:", X_test.shape)

        feature_weights = np.ones(X_train.shape[1])
        for i, feature in enumerate(features):
            if "srch_rqst" in feature:
                feature_weights[i] = 2.0
            elif feature == "hour" or "hour_" in feature:
                feature_weights[i] = 1.5
            elif "conversion_rate" in feature or "completion_rate" in feature:
                feature_weights[i] = 1.3
            elif "cancellation_rate" in feature:
                feature_weights[i] = 1.2

        X_train_weighted = X_train * feature_weights

        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=0.7,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_weighted, y_train)

        if len(X_val) > 0:
            y_pred_val = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            r2 = r2_score(y_val, y_pred_val)
            print(f"Validation RMSE: {rmse:.4f}")
            print(f"Validation R-squared: {r2:.4f}")

        y_pred_test = model.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_test = r2_score(y_test, y_pred_test)
        print(f"Test RMSE: {rmse_test:.4f}")
        print(f"Test R-squared: {r2_test:.4f}")

        feature_importance = pd.DataFrame(
            {"Feature": features, "Importance": model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        print("\nTop 10 important features:")
        print(feature_importance.head(10))

        return model, features

    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        raise


def predict_and_recommend(
    model,
    features,
    user_time,
    user_location,
    user_vehicle_type,
    all_wards,
    df,
    original_df,
):

    all_wards = sorted(list(map(str, all_wards)))
    predictions = []

    user_time_dt = pd.to_datetime(user_time)
    current_hour = user_time_dt.hour
    day_of_week = user_time_dt.dayofweek
    day_of_month = user_time_dt.day
    month = user_time_dt.month
    is_weekend = 1 if day_of_week >= 5 else 0

    morning_hour = 1 if 6 <= current_hour <= 10 else 0
    evening_hour = 1 if 16 <= current_hour <= 20 else 0
    night_hour = 1 if current_hour >= 21 or current_hour <= 5 else 0
    lunch_hour = 1 if 11 <= current_hour <= 15 else 0
    hour_squared = current_hour**2
    hour_sin = np.sin(2 * np.pi * current_hour / 24)
    hour_cos = np.cos(2 * np.pi * current_hour / 24)

    india_holidays = holidays.India()
    is_holiday = 1 if user_time_dt.date() in india_holidays else 0

    is_peak_hour = 1 if (7 <= current_hour <= 9) or (17 <= current_hour <= 19) else 0

    ward_stats = {}
    for ward in all_wards:
        ward_data = original_df[original_df["ward_num"] == ward]
        if not ward_data.empty:

            ward_stats[ward] = {
                "avg_srch_rqst": ward_data.get("srch_rqst", pd.Series([0])).mean(),
                "avg_drivers": ward_data.get("active_drvr", pd.Series([0])).mean(),
                "avg_booking_rate": (
                    (
                        ward_data.get("booking", pd.Series([0])).mean()
                        / (ward_data.get("srch_rqst", pd.Series([0])).mean() + 1e-6)
                    )
                    if "booking" in ward_data.columns
                    and "srch_rqst" in ward_data.columns
                    else 0
                ),
                "avg_conversion_rate": ward_data.get(
                    "conversion_rate", pd.Series([0])
                ).mean(),
                "avg_completion_rate": ward_data.get(
                    "completion_rate", pd.Series([0])
                ).mean(),
                "avg_cancellation_rate": ward_data.get(
                    "cancellation_rate", pd.Series([0])
                ).mean(),
                "avg_earning_per_ride": ward_data.get(
                    "avg_earning_per_ride", pd.Series([0])
                ).mean(),
                "avg_distance": ward_data.get("avg_distance", pd.Series([0])).mean(),
                "rider_satisfaction": ward_data.get(
                    "rider_satisfaction_proxy", pd.Series([0])
                ).mean(),
                "queue_effectiveness": ward_data.get(
                    "queue_effectiveness", pd.Series([0])
                ).mean(),
            }
        else:
            ward_stats[ward] = {
                "avg_srch_rqst": 0,
                "avg_drivers": 0,
                "avg_booking_rate": 0,
                "avg_conversion_rate": 0.5,
                "avg_completion_rate": 0.5,
                "avg_cancellation_rate": 0.5,
                "avg_earning_per_ride": 0,
                "avg_distance": 0,
                "rider_satisfaction": 0.5,
                "queue_effectiveness": 0.5,
            }

    recent_data = original_df[original_df["hour"] == current_hour].copy()
    recent_hour_stats = {}
    for ward in all_wards:
        ward_hour_data = recent_data[recent_data["ward_num"] == ward]
        if not ward_hour_data.empty:
            recent_hour_stats[ward] = {
                "recent_srch_rqst": ward_hour_data.get(
                    "srch_rqst", pd.Series([0])
                ).mean(),
                "recent_drivers": ward_hour_data.get(
                    "active_drvr", pd.Series([0])
                ).mean(),
                "recent_conversion_rate": ward_hour_data.get(
                    "conversion_rate", pd.Series([0])
                ).mean(),
                "recent_completion_rate": ward_hour_data.get(
                    "completion_rate", pd.Series([0])
                ).mean(),
                "recent_cancellation_rate": ward_hour_data.get(
                    "cancellation_rate", pd.Series([0])
                ).mean(),
            }
        else:
            recent_hour_stats[ward] = {
                "recent_srch_rqst": 0,
                "recent_drivers": 0,
                "recent_conversion_rate": 0.5,
                "recent_completion_rate": 0.5,
                "recent_cancellation_rate": 0.5,
            }

    for ward in all_wards:
        input_data = {}

        for feature in features:
            input_data[feature] = 0

        input_data["hour"] = current_hour
        input_data["day_of_week"] = day_of_week
        input_data["day_of_month"] = day_of_month
        input_data["month"] = month
        input_data["is_weekend"] = is_weekend
        input_data["is_holiday"] = is_holiday
        input_data["is_peak_hour"] = is_peak_hour

        input_data["morning_hour"] = morning_hour
        input_data["evening_hour"] = evening_hour
        input_data["night_hour"] = night_hour
        input_data["lunch_hour"] = lunch_hour
        input_data["hour_squared"] = hour_squared
        input_data["hour_sin"] = hour_sin
        input_data["hour_cos"] = hour_cos

        ward_feature = f"ward_num_{ward}"
        if ward_feature in input_data:
            input_data[ward_feature] = 1

        vehicle_feature = f"vehicle_type_{user_vehicle_type}"
        if vehicle_feature in input_data:
            input_data[vehicle_feature] = 1

        avg_search = ward_stats[ward]["avg_srch_rqst"]
        if "srch_rqst" in input_data:
            input_data["srch_rqst"] = avg_search

        if "srch_rqst_log" in input_data:
            input_data["srch_rqst_log"] = np.log1p(avg_search)
        if "srch_rqst_scaled" in input_data:
            input_data["srch_rqst_scaled"] = avg_search / 100

        if "srch_hour_interaction" in input_data:
            input_data["srch_hour_interaction"] = avg_search * current_hour
        if "srch_weekend_interaction" in input_data:
            input_data["srch_weekend_interaction"] = avg_search * is_weekend

        for feature in features:
            if "lag" in feature and feature in input_data:
                if "ward_" in feature and ward in feature:
                    input_data[feature] = avg_search
                elif "lag_1h" in feature:
                    input_data[feature] = avg_search
                elif "lag_2h" in feature:
                    input_data[feature] = avg_search * 0.9
                elif "lag_1d" in feature:
                    input_data[feature] = avg_search * 1.1

        if "srch_rqst_rolling_3h" in input_data:
            input_data["srch_rqst_rolling_3h"] = avg_search
        if "srch_rqst_rolling_6h" in input_data:
            input_data["srch_rqst_rolling_6h"] = avg_search

        input_df = pd.DataFrame([input_data])

        try:
            demand_prediction = model.predict(input_df)[0]

            current_distance = 0
            if user_location == ward:
                current_distance = 0
            else:
                current_distance = (
                    abs(int(ward) - int(user_location))
                    if ward.isdigit() and user_location.isdigit()
                    else 5
                )

            demand_weight = 0.3
            search_weight = 0.2
            distance_weight = 0.15
            driver_weight = 0.1
            conversion_weight = 0.15
            cancellation_weight = 0.1

            recent_stats = recent_hour_stats[ward]
            ward_stats_data = ward_stats[ward]

            max_distance = 10
            normalized_distance = 1 - (
                min(current_distance, max_distance) / max_distance
            )

            max_drivers = max(
                1, max([s["recent_drivers"] for s in recent_hour_stats.values()])
            )
            normalized_drivers = 1 - (
                min(recent_stats["recent_drivers"], max_drivers) / max_drivers
            )

            max_search = max(
                1, max([s["recent_srch_rqst"] for s in recent_hour_stats.values()])
            )
            normalized_search = (
                min(recent_stats["recent_srch_rqst"], max_search) / max_search
            )

            result = {
                "ward": ward,
                "predicted_demand": demand_prediction,
                "distance": current_distance,
                "search_requests": recent_stats["recent_srch_rqst"],
                "current_drivers": recent_stats["recent_drivers"],
                "conversion_rate": recent_stats["recent_conversion_rate"],
                "completion_rate": recent_stats["recent_completion_rate"],
                "cancellation_rate": recent_stats["recent_cancellation_rate"],
                "avg_earning_per_ride": ward_stats_data["avg_earning_per_ride"],
                "rider_satisfaction": ward_stats_data["rider_satisfaction"],
                "queue_effectiveness": ward_stats_data["queue_effectiveness"],
                "score": 0,
            }

            result["score"] = (
                demand_weight * (demand_prediction / 100)
                + search_weight * normalized_search
                + distance_weight * normalized_distance
                + driver_weight * normalized_drivers
                + conversion_weight * recent_stats["recent_conversion_rate"]
                + cancellation_weight
                * (1 - min(recent_stats["recent_cancellation_rate"], 1))
            )

            predictions.append(result)

        except Exception as e:
            print(f"Error predicting for ward {ward}: {e}")

    predictions.sort(key=lambda x: x["score"], reverse=True)
    top_n = 3

    recommendations = predictions[:top_n]
    return recommendations


def main(
    trends_file, driver_file, funnel_file, user_time, user_location, user_vehicle_type
):
    try:

        merged_df = load_and_preprocess_data(trends_file, driver_file, funnel_file)

        df_with_features = create_features(merged_df)

        print("\nData Overview:")
        print(
            f"Date range: {df_with_features['date'].min()} to {df_with_features['date'].max()}"
        )
        print(
            f"Hour range: {df_with_features['hour'].min()} to {df_with_features['hour'].max()}"
        )
        print(f"Number of unique wards: {merged_df['ward_num'].nunique()}")
        print(f"Number of vehicle types: {merged_df['vehicle_type'].nunique()}")
        print("\nAvailable columns:", df_with_features.columns.tolist())

        model, features = train_model(df_with_features)

        all_wards = sorted(merged_df["ward_num"].unique())

        recommendations = predict_and_recommend(
            model,
            features,
            user_time,
            user_location,
            user_vehicle_type,
            all_wards,
            df_with_features,
            merged_df,
        )

        print("\n--- High Demand Area Recommendations ---")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. Ward {rec['ward']}:")
            print(f"   - Search Requests: {rec['search_requests']:.0f}")
            print(f"   - Distance from current location: {rec['distance']}")
            print(f"   - Current active drivers: {rec['current_drivers']:.0f}")
            print(f"   - Cancellation Rate: {rec['cancellation_rate']:.2%}")
            print(f"   - Avg Earning per Ride: ₹{rec['avg_earning_per_ride']:.2f}")
            print(f"   - Rider Satisfaction: {rec['rider_satisfaction']:.2f}")

        return recommendations, model, features

    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    trends_file = "./data/chennai/trends_live_ward_new_key.json"
    driver_file = "./data/chennai/driver_eda_wards_new_key.json"
    funnel_file = "./data/chennai/funnel_live_ward_new_key.json"

    user_time = "2025-03-13 10:00:00"
    user_location = "174"
    user_vehicle_type = "Auto"

    main(
        trends_file,
        driver_file,
        funnel_file,
        user_time,
        user_location,
        user_vehicle_type,
    )
