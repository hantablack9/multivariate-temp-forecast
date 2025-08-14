# # In src/pipelines/02_feature_engineering.py

import os
import sys
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional
import joblib  # Import joblib to save the scaler
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Add root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.multivariate_temp_forecast.logger_factory import ObservationLogger


# 1. Load the configuration from the YAML file
with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

@dataclass
class FeatureEngineer:
    """
    A class to handle the feature engineering and scaling process.
    It loads preprocessed data, applies feature engineering and scaling,
    and saves the final dataset AND the fitted scaler.

    1. Load the preprocessed data.
    2. Apply scaling.
    3. Apply feature engineering (windowing and feature_selection).
    4. Save scaler and feature engineered data.
    """
    # --- Configuration Parameters ---
    # Every parameter controlled by params.yaml MUST be a dataclass field with a type hint
    window_size: int = 5
    train_test_split: float = 0.75
    test_eval_split: float = 0.50
    period: Literal['1h', '1d', '6h', '6d'] = field(default='1d', metadata={"help": "The window period for feature engineering."})
    scaler_type: Literal['minmax', 'standard'] = field(default='minmax', metadata={"help": "The type of scaler to use."})
    
    # --- Class Constants ---
    PREPROCESSED_CSV_DIR: ClassVar[str] = os.path.join("data", "preprocessed")
    PREPROCESSED_CSV_PATH: ClassVar[str] = os.path.join(PREPROCESSED_CSV_DIR, "preprocessed.csv")
    FEATURE_ENGINEERED_CSV_DIR: ClassVar[str] = os.path.join("data", "feature_engineered")
    FEATURE_ENGINEERED_CSV_PATH: ClassVar[str] = os.path.join(FEATURE_ENGINEERED_CSV_DIR, "feature_engineered.csv")
    # --- NEW: Get the path to save the scaler ---
    SCALER_DIR: ClassVar[str] = os.path.join(FEATURE_ENGINEERED_CSV_DIR, "scaler")

    def __post_init__(self):
        # Create directories if they don't exist
        os.makedirs(self.FEATURE_ENGINEERED_CSV_DIR, exist_ok=True)
        os.makedirs(self.SCALER_DIR, exist_ok=True)
        self.logger = ObservationLogger(log_file="feature_engineer.log", tag = "feature_engineer")

    def 

    def windower(self, df, window_size):
        return df.rolling(window=window_size, min_periods=1).mean()




# def feature_engineer(config_path: str):
#     """
#     Loads preprocessed data, applies feature engineering and scaling,
#     and saves the final dataset AND the fitted scaler.
#     """
#     print("--- Starting Feature Engineering & Scaling Stage ---")

#     # Load configuration
#     with open(config_path) as f:
#         config = yaml.safe_load(f)

#     input_path = config["preprocess"]["outs"][0]
#     output_data_path = config["feature_engineer"]["outs"][0]
#     # --- NEW: Get the path to save the scaler ---
#     output_scaler_path = config["feature_engineer"]["outs"][1]

#     # Load the preprocessed data
#     print(f"Loading data from: {input_path}")
#     df = pd.read_csv(input_path, parse_dates=["timestamp"])

#     # 1. Fix wind anomalies and encode features
#     print("Fixing anomalies and encoding features...")
#     df = fix_wind_anomalies(df)
#     df = encode_sinusoidal_features(df)

#     # --- NEW: Data Scaling ---
#     print("Scaling data...")

#     # Select only numeric columns for scaling (leave timestamp and identifiers)
#     numeric_cols = df.select_dtypes(include="number").columns.tolist()

#     # Choose scaler from config
#     scaler_type = config["feature_engineering"]["scaler"]
#     if scaler_type == "standard":
#         scaler = StandardScaler()
#     elif scaler_type == "minmax":
#         scaler = MinMaxScaler()
#     else:
#         raise ValueError(f"Unsupported scaler type: {scaler_type}")

#     # Fit the scaler ONLY on the training data portion (important!)
#     # Assuming a simple time-based split for this example.
#     # A real implementation might have a more robust train/test split definition.
#     train_end_date = pd.to_datetime(config["data_split"]["train_end_date"])
#     train_df = df[df["timestamp"] <= train_end_date]

#     print(f"Fitting scaler on data up to {train_end_date}...")
#     scaler.fit(train_df[numeric_cols])

#     # Transform the entire dataset using the fitted scaler
#     df[numeric_cols] = scaler.transform(df[numeric_cols])

#     # --- NEW: Save the fitted scaler ---
#     scaler_dir = os.path.dirname(output_scaler_path)
#     os.makedirs(scaler_dir, exist_ok=True)
#     print(f"Saving scaler to: {output_scaler_path}")
#     joblib.dump(scaler, output_scaler_path)
#     # --- END OF NEW STEPS ---

#     # Save the scaled feature-engineered dataset
#     os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
#     print(f"Saving scaled features to: {output_data_path}")
#     df.to_csv(output_data_path, index=False)

#     print("--- Feature Engineering & Scaling Stage Complete ---")


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", dest="config_path", required=True, help="Path to the config file")
#     args = parser.parse_args()

#     feature_engineer(args.config_path)


# from dataclasses import dataclass
# from typing import Literal, Optional
# import pandas as pd
# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin
# # Assuming logger_factory.py is in a reachable path
# from src.logger_factory import ObservationLogger


# class FeatureEngineer(BaseEstimator, TransformerMixin):
#     def __init__(self, datetime_col="Timestamp", scale_type="standard", scale_cols=None, cyclic_cols: Optional[list[str]]=None):
#         self.datetime_col = datetime_col
#         self.scale_type = scale_type
#         self.scale_cols = scale_cols or []
#         self.scaler = None
#         self.cyclic_cols = cyclic_cols
#         self.scale_params = {}

#     def fit(self, X, y=None):
#         df = X.copy()
#         self.scale_params = {}
#         if self.scale_type == "standard":
#             for col in self.scale_cols:
#                 self.scale_params[col] = {"mean": df[col].mean(), "std": df[col].std()}
#         elif self.scale_type == "minmax":
#             for col in self.scale_cols:
#                 self.scale_params[col] = {"min": df[col].min(), "max": df[col].max()}
#         return self

#     def transform(self, X):
#         df = X.copy()

#         for col in self.scale_cols:
#             if self.scale_type == "standard":
#                 mean, std = self.scale_params[col]["mean"], self.scale_params[col]["std"]
#                 df[f"sc_{col}"] = (df[col] - mean) / std
#             elif self.scale_type == "minmax":
#                 min_, max_ = self.scale_params[col]["min"], self.scale_params[col]["max"]
#                 df[f"mm_{col}"] = (df[col] - min_) / (max_ - min_)

#         return df

#     def inverse_transform(self, df_scaled):
#         df = df_scaled.copy()
#         for col in self.scale_cols:
#             if self.scale_type == "standard":
#                 mean, std = self.scale_params[col]["mean"], self.scale_params[col]["std"]
#                 df[col] = df[f"sc_{col}"] * std + mean
#             elif self.scale_type == "minmax":
#                 min_, max_ = self.scale_params[col]["min"], self.scale_params[col]["max"]
#                 df[col] = df[f"mm_{col}"] * (max_ - min_) + min_
#         return df
