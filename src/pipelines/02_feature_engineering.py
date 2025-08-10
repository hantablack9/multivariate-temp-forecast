# In src/pipelines/02_feature_engineering.py

import os

import joblib  # Import joblib to save the scaler
import pandas as pd
import yaml

# Import your other feature engineering functions
from src.multivariate_temp_forecasting.feature_engineering import (
    encode_sinusoidal_features,
    fix_wind_anomalies,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def feature_engineer(config_path: str):
    """
    Loads preprocessed data, applies feature engineering and scaling,
    and saves the final dataset AND the fitted scaler.
    """
    print("--- Starting Feature Engineering & Scaling Stage ---")

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    input_path = config["preprocess"]["outs"][0]
    output_data_path = config["feature_engineer"]["outs"][0]
    # --- NEW: Get the path to save the scaler ---
    output_scaler_path = config["feature_engineer"]["outs"][1]

    # Load the preprocessed data
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path, parse_dates=["timestamp"])

    # 1. Fix wind anomalies and encode features
    print("Fixing anomalies and encoding features...")
    df = fix_wind_anomalies(df)
    df = encode_sinusoidal_features(df)

    # --- NEW: Data Scaling ---
    print("Scaling data...")

    # Select only numeric columns for scaling (leave timestamp and identifiers)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Choose scaler from config
    scaler_type = config["feature_engineering"]["scaler"]
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")

    # Fit the scaler ONLY on the training data portion (important!)
    # Assuming a simple time-based split for this example.
    # A real implementation might have a more robust train/test split definition.
    train_end_date = pd.to_datetime(config["data_split"]["train_end_date"])
    train_df = df[df["timestamp"] <= train_end_date]

    print(f"Fitting scaler on data up to {train_end_date}...")
    scaler.fit(train_df[numeric_cols])

    # Transform the entire dataset using the fitted scaler
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # --- NEW: Save the fitted scaler ---
    scaler_dir = os.path.dirname(output_scaler_path)
    os.makedirs(scaler_dir, exist_ok=True)
    print(f"Saving scaler to: {output_scaler_path}")
    joblib.dump(scaler, output_scaler_path)
    # --- END OF NEW STEPS ---

    # Save the scaled feature-engineered dataset
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    print(f"Saving scaled features to: {output_data_path}")
    df.to_csv(output_data_path, index=False)

    print("--- Feature Engineering & Scaling Stage Complete ---")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", required=True, help="Path to the config file")
    args = parser.parse_args()

    feature_engineer(args.config_path)
