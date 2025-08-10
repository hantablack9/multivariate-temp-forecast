# In src/pipelines/01_preprocess.py

import yaml

from src.multivariate_temp_forecast.data_etl import DataETL

# 1. Load the configuration from the YAML file
with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# 2. Instantiate your class, passing the loaded parameters.
# The ** operator unpacks the dictionary, overriding the defaults in your class.
etl_processor = DataETL(**params["data_etl"])

# 3. Run the pipeline
etl_processor.run(impute=True, resample=True, save=True)

print("Preprocessing stage complete.")


# from dataclasses import dataclass
# from typing import Literal, Optional
# import pandas as pd
# import numpy as np
# from calendar import isleap
# from sklearn.base import BaseEstimator, TransformerMixin
# # Assuming logger_factory.py is in a reachable path
# from src.logger_factory import ObservationLogger


# @dataclass
# class SinusoidalEncoder:
#     """Encodes datetime and cyclic columns into sinusoidal features.
#     Attributes
#     ----------
#     datetime_col : str
#         The name of the datetime column in the DataFrame.
#     cyclic_cols : Optional[list[str]]
#         The names of the cyclic columns in the DataFrame.
#     Methods
#     ----------
#     transform(df: pd.DataFrame) -> pd.DataFrame
#         Encodes datetime and cyclic columns into sinusoidal features.
#     Parameters
#     ----------
#     df : pd.DataFrame
#         The DataFrame to be transformed.
#     Returns
#     -------
#     pd.DataFrame
#         The DataFrame with sinusoidal features added.
#     Raises
#     ------
#     ValueError
#         If the 'datetime' column is not found in the DataFrame.
#     Examples
#     --------
#     >>> sinusoidal_encoder = SinusoidalEncoder(datetime_col="datetime", cyclic_cols=["wd (deg)"])
#     >>> df_transformed = sinusoidal_encoder.transform(df)
#     >>> print(df_transformed.head())
#         datetime  wd (deg)  hour_sin  hour_cos  day_sin  day_cos  month_sin  month_cos
#         0 2020-01-01 00:00:00       180  0.000000  1.000000  0.000000  1.000000  0.500000 -0.866025
#         1 2020-01-01 00:10:00       180  0.000000  1.000000  0.000000  0.500000 -0.866025  0.000000  1.000000
#         2 2020-01-01 00:20:00       180  0.000000  1.000000  0.000000  0.500000 -0.866025  0.000000  1.000000
#         3 2020-01-01 00:30:00       180  0.000000  1.000000  0.000000  0.500000 -0.866025  0.000000  1.000000
#         4 2020-01-01 00:40:00       180  0.000000  1.000000  0.000000  0.500000 -0.866025  0.000000  1.000000
#     """

#     datetime_col: str = "datetime"
#     cyclic_cols: Optional[list[str]] = None

#     def __init__(self, datetime_col: str = "datetime", cyclic_cols: Optional[list[str]] = None):
#         """
#         Initializes the SinusoidalEncoder with the specified datetime column and cyclic columns.

#         Args:
#             datetime_col (str): The name of the datetime column in the DataFrame.
#             cyclic_cols (Optional[list[str]]): The names of the cyclic columns in the DataFrame.
#         """
#         self.datetime_col = datetime_col
#         self.cyclic_cols = cyclic_cols if cyclic_cols is not None else []
#         if not isinstance(self.cyclic_cols, list):
#             raise CyclicalEncodingError("cyclic_cols must be a list of column names.")
#         if not all(isinstance(col, str) for col in self.cyclic_cols):
#             raise CyclicalEncodingError("All cyclic_cols must be strings representing column names.")
#     def day_encoder(
#         self, date: Optional[pd.Timestamp]
#     ) -> Optional[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
#         """
#         Returns sinusoidal encoding (sin, cos) for hour, day, and month of a timestamp.

#         Args:
#             date (Optional[pd.Timestamp]): The datetime value to encode.

#         Returns:
#             Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
#             Tuple of sin/cos pairs for hour, day, and month. Returns None if input is NaT.
#         """
#         if pd.isnull(date):
#             return None

#         def days_in_month(year: int, month: int) -> int:
#             """
#             Returns the number of days in a given month of a year, accounting for leap years.

#             Args:
#                 year (int): The year (used to check leap year for February).
#                 month (int): The month (1 to 12).

#             Returns:
#                 int: Number of days in the given month.
#             """
#             month_days = {
#                 1: 31,
#                 2: 29 if isleap(year) else 28,
#                 3: 31,
#                 4: 30,
#                 5: 31,
#                 6: 30,
#                 7: 31,
#                 8: 31,
#                 9: 30,
#                 10: 31,
#                 11: 30,
#                 12: 31,
#             }
#             return month_days[month]

#         days = days_in_month(date.dt.year, date.dt.month)

#         hour_sin = np.sin(2 * np.pi * date.dt.hour / 24)
#         hour_cos = np.cos(2 * np.pi * date.dt.hour / 24)

#         day_sin = np.sin(2 * np.pi * date.dt.day / days)
#         day_cos = np.cos(2 * np.pi * date.dt.day / days)

#         month_sin = np.sin(2 * np.pi * date.dt.month / 12)
#         month_cos = np.cos(2 * np.pi * date.dt.month / 12)

#         return (hour_sin, hour_cos), (day_sin, day_cos), (month_sin, month_cos)

#     def transform(
#         self, df: pd.DataFrame, datetime_col: str = "datetime", cyclic_cols: Optional[list[str]] = None
#     ) -> pd.DataFrame:
#         """
#         Encodes datetime and cyclic columns into sinusoidal features.
#         Parameters
#         ----------
#         df : pd.DataFrame
#             The DataFrame to be transformed.
#         Returns
#         -------
#         pd.DataFrame
#             The DataFrame with sinusoidal features added.
#         """
#         if datetime_col in df.columns:
#             ts = pd.to_datetime(df[datetime_col], errors="coerce")
#             df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
#             df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
#             df["day_sin"] = np.sin(2 * np.pi * ts.dt.day / 31)
#             df["day_cos"] = np.cos(2 * np.pi * ts.dt.day / 31)
#             df["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
#             df["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)
#         else:
#             raise  UnsupportedColumnError()
#         if df[cyclic_cols].isnull().any().any() and cyclic_cols:
#             for col in cyclic_cols:
#                 df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / 360)
#                 df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / 360)

#         return df


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
