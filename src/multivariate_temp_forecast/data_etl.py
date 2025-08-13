"""
DVC compatible ETL script to load and process the Jena Climate dataset.

Preprocesses the data by imputing missing values, resampling data,
and filtering by year range.

Climate Data Time-Series
We will be using Jena Climate dataset recorded by the Max Planck
Institute for Biogeochemistry. The dataset consists of 14 features
such as temperature, pressure, humidity etc, recorded once per 10
minutes.

Location: Weather Station, Max Planck Institute for Biogeochemistry in Jena, Germany

Time-frame Considered: Jan 10, 2009 - December 31, 2016

Usage:
    python src/components/data_etl.py

Author: Hanish Paturi
"""

import os
import shutil

# import json
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional

import keras
import numpy as np
import pandas as pd

from src.multivariate_temp_forecast.logger_factory import ObservationLogger

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class DataProcessingError(Exception):
    """Custom exception for errors during the ETL pipeline execution.

    Attributes:
        message (str): The error message.
    """

    # --- FIX for TRY003 ---
    # Define standard messages inside the class
    NO_DATA_TO_SAVE = "No data to save. Please run the pipeline first."
    EXTRACTION_FAILED = "Failed to extract or read the dataset"
    # --- END OF FIX ---


class InvalidParametersError(Exception):
    """Custom exception for invalid parameter values."""

    IMPUTE_NOT_PROVIDED = "Invalid request: Imputation is \
        mandatory for resampling or filtering."
    IMPUTATION_METHODS = "Valid imputation values: ['seasonal', 'forward_fill', \
        'rolling_mean', 'rolling_median']"
    RESAMPLE_METHODS = "Valid resampling values: ['mean', 'median', 'first', 'last']"
    SAMPLING_RATES = "Valid sampling values: ['1H', '1D', '1M', '1Y']"
    FILTER_YEARS = "Valid filter years are: 'True', 'False'"
    FILTER_RANGE = "Filter years range must be in inclusive range of [2009, 2016]"
    FILTER_COUNT = "Filter range must contain exactly two years."
    FILTER_BOUNDS = "First bound must be less than second bound"
    INDEX_DTYPE = "Index must be a DatetimeIndex to filter by year"


class SinusoidalEncodingError(Exception):
    """Custom exception for errors during the cyclical encoding process."""

    CYCLIC_COLUMNS_NOT_FOUND = "Cyclic columns not found in the DataFrame"
    DATETIME_COLUMN_NOT_FOUND = "Datetime column not found in the DataFrame"
    DATETIME_INDEX_NOT_FOUND = "Datetime index not found in the DataFrame"
    DATETIME_TYPE_ERROR = "Datetime index must be a DatetimeIndex type"


@dataclass
class DataETL:
    """A DVC-compatible class to manage the ETL process for the Jena Climate dataset.

    The class is composed of several private methods to handle each step of the ETL process,
    orchestrated by the main `run()` method.

    This class is designed to work within a Data Version Control (DVC) pipeline.
    It handles the extraction, cleaning, imputation, and transformation of the data,
    producing artifacts in fixed, predictable locations. The state of the pipeline
    (e.g., configuration parameters) is intended to be versioned in a `params.yaml`
    file, while the output data artifacts are versioned by DVC.

    The class enforces a strict operational order:
        1. Data extraction and loading.
        2. Data imputation (optional).
        3. Data resampling (optional).
        4. Data filtering (optional).

    Note: Imputation is mandatory for any subsequent resampling or filtering steps to
        ensure data integrity.

    The class provides a logger for tracking observations with optional metadata.
    It supports various formats including plain text, SQLite, CSV, JSON, and Markdown.
    It is designed to be extensible and can be integrated into larger data processing pipelines.
    The logger can be used to log events, errors, and general notes in a structured manner.
    It also supports grouping observations by date and filtering by tags, sections, and time ranges.
    It is suitable for data science projects, especially those involving time series analysis and
    machine learning.

    Attributes:
        imputation_method (str): The strategy to use for filling missing values.
        resample_method (str): The aggregation method for resampling (e.g., 'mean').
        sampling_rate (str): The target frequency for resampling (e.g., '1H').
        filter_years (bool): If True, filters the data to the specified year range.
        filter_range (List[int]): The start and end years for filtering.
        processed_df (pd.DataFrame): Holds the DataFrame after the last successful run.

    Example:
        # Load parameters from a YAML file
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load()

        # Instantiate the processor with the loaded parameters
        etl_processor = DataETL(**params['etl'])

        # Run the pipeline based on run-time flags
        etl_processor.run(**params['run'])

    Methods:
        run: Executes the ETL pipeline based on the provided flags.
        _extract_and_load: Extracts the raw data from the source and loads it into a DataFrame.
        _fix_wind_speed_errors: Fixes wind speed errors in the DataFrame.
        _impute_data: Imputes missing values in the DataFrame.
        _resample_data: Resamples the DataFrame to the specified frequency.
        _filter_data: Filters the DataFrame to the specified year range.
        _encode_sinusoidal: Encodes cyclic data using a Sinusoidal encoding.
    """

    # --- Configuration Parameters ---
    # Every parameter controlled by params.yaml MUST be a dataclass field with a type hint
    impute: bool = True
    resample: bool = True
    filter_years: bool = False
    fix_data_errors: bool = True
    imputation_method: Literal["seasonal", "forward_fill", "rolling_mean", "rolling_median"] = "seasonal"
    resample_method: Literal["mean", "median", "first", "last"] = "mean"
    sampling_rate: Literal["1h", "1d", "1y"] = "1h"
    filter_range: list[int] = field(default_factory=lambda: [2014, 2017], repr=False)
    encode_sinusoidal: bool = True
    cyclic_cols: list[str] = field(default_factory=lambda: ["wd_deg"], repr=False)
    anomaly_cols: list[str] = field(default_factory=lambda: ["wv_ms", "max_wv_ms"], repr=False)

    # --- Internal State ---
    raw_data_exists: bool = field(init=False, repr=False)
    processed_df: Optional[pd.DataFrame] = field(init=False, default=None, repr=False)

    # --- Class Constants ---
    RAW_DATA_DIR: ClassVar[str] = os.path.join("data", "raw")
    PREPROCESSED_CSV_DIR: ClassVar[str] = os.path.join("data", "preprocessed")
    RAW_CSV_PATH: ClassVar[str] = os.path.join(RAW_DATA_DIR, "jena_climate_2009_2016.csv")
    PREPROCESSED_CSV_PATH: ClassVar[str] = os.path.join(PREPROCESSED_CSV_DIR, "preprocessed.csv")

    def __post_init__(self):
        """Initializes the logger and ensures data directories exist."""
        self.logger = ObservationLogger(log_file="data_etl.log")
        self.logger.update_tags({
            "etl": "Main ETL Process Control",
            "extract": "Data Extraction Step",
            "load": "Data Loading & Cleaning Step",
            "encode": "Cyclic Encoding Step",
            "impute": "Imputation Step",
            "filter": "Filtering Step",
            "resample": "Resampling Step",
            "error": "Error Messages",
            "validation": "Input validation",
            "io": "File Input/Output",
            "info": "Informational messages",
            "fix": "Data Fixing Step",
        })
        # # The check happens here, when the object is created.
        # # Moving this to the idempotent _download_if_needed().
        # self.raw_data_exists = os.path.exists(self.RAW_CSV_PATH)
        # os.makedirs(self.RAW_DATA_DIR, exist_ok=True)
        # os.makedirs(self.PREPROCESSED_CSV_DIR, exist_ok=True)

        self._validate_parameters()

    # --- NEW VALIDATION LOGIC ---
    def _validate_parameters(self):
        """Private method to validate configuration parameters."""
        self.logger.log("Validating configuration parameters.", tag=["validation", "info"])

        if not self.impute and (self.resample or self.filter_years):
            error_msg = "Invalid request: Imputation is mandatory for resampling or filtering."
            self.logger.log(error_msg, tag=["validation", "error"], level="warning")
            raise InvalidParametersError(InvalidParametersError.IMPUTE_NOT_PROVIDED)

        if self.imputation_method not in ["seasonal", "forward_fill", "rolling_mean", "rolling_median"]:
            error_msg = f"{InvalidParametersError.IMPUTATION_METHODS}, got {self.imputation_method}"
            self.logger.log(error_msg, tag=["validation", "error"], level="warning")
            raise InvalidParametersError(InvalidParametersError.IMPUTATION_METHODS)
        # Check if filter_range has exactly two values
        if self.filter_years:
            if len(self.filter_range) != 2:
                error_msg = f"{InvalidParametersError.FILTER_COUNT}, got {self.filter_range}"
                self.logger.log(error_msg, tag=["validation", "error"], level="warning")
                raise InvalidParametersError(error_msg)

            start_year, end_year = self.filter_range
            # Check if the years are within the allowed bounds (2009-2016)
            if not (2009 <= start_year <= 2016 and 2009 <= end_year <= 2016):
                error_msg = f"{InvalidParametersError.FILTER_RANGE}, got {self.filter_range}"
                self.logger.log(error_msg, tag=["validation", "error"], level="warning")
                raise InvalidParametersError(error_msg)
            # Check if the start year is before the end year
            if start_year >= end_year:
                error_msg = f"{InvalidParametersError.FILTER_BOUNDS}, got {self.filter_range}"
                self.logger.log(error_msg, tag=["validation", "error"], level="warning")
                raise InvalidParametersError(error_msg)
            self.logger.log("Configuration parameters are valid.", tag=["validation", "info"])

    # --- END OF NEW VALIDATION LOGIC ---

    # --- NEW: Dedicated Download Method ---
    def download_if_needed(self):
        """
        Checks for the raw data file and downloads it if it does not exist.
        This method is idempotent and safe to call multiple times.
        """
        # Check for the file at the moment it's needed.
        if os.path.exists(self.RAW_CSV_PATH):
            self.logger.log("Raw data already exists. Skipping download.", tag="extract")
            return

        self.logger.log("Raw data not found. Starting download.", tag="extract")

        # Ensure the target directory exists before moving the file.
        os.makedirs(self.RAW_DATA_DIR, exist_ok=True)

        temp_cache_dir = "temp_cache"
        try:
            os.makedirs(temp_cache_dir, exist_ok=True)
            zip_path = keras.utils.get_file(
                origin= "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
                fname = "jena_climate_2009_2016.zip",
                extract = False,
                cache_dir=os.path.abspath(temp_cache_dir )
            )
            shutil.unpack_archive(zip_path, temp_cache_dir)
            extracted_csv_path = os.path.join(temp_cache_dir, "jena_climate_2009_2016.csv")
            shutil.move(extracted_csv_path, self.RAW_CSV_PATH)
            self.logger.log(f"Raw data successfully saved to {self.RAW_CSV_PATH}", tag="extract")
        except Exception as e:
            raise DataProcessingError(DataProcessingError.EXTRACTION_FAILED) from e
        finally:
            if os.path.exists(temp_cache_dir):
                shutil.rmtree(temp_cache_dir)
    # --- END OF NEW METHOD ---

    def _extract_and_load(self) -> pd.DataFrame:
        """Extracts the dataset and performs initial cleaning and preparation.

        Ensures raw data is present by calling the download helper, then loads and
        cleans the data from the local CSV file.

        This method performs the following operations: extracting and loading the CSV,
        standardizing column names, parsing the datetime column, and reindexing
        to a consistent 10-minute frequency, which introduces NaNs for DVC to track.

        Returns:
            pd.DataFrame: The raw, cleaned DataFrame with a DatetimeIndex.

        Raises:
            DataProcessingError: If the data cannot be downloaded, extracted, or read.
        """
        # --- THIS IS THE REFACTORED LOGIC ---
        # 1. Ensure the data is present.
        self.download_if_needed()
        # --- END OF REFACTORED LOGIC ---

        self.logger.log(f"Loading and cleaning data from {self.RAW_CSV_PATH}", tag="load")
        df = pd.read_csv(self.RAW_CSV_PATH)

        # df = pd.read_csv(self.RAW_CSV_PATH)
        df.columns = (
            df.columns.str.replace(r"\s*\([^)]*\)", "", regex=True)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(".", "_", regex=False)
        )
        df["date_time"] = pd.to_datetime(df["date_time"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
        df = df.drop_duplicates(subset=["date_time"]).sort_values("date_time").set_index("date_time")

        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="10min")
        df = df.reindex(full_index)
        df.index.name = "timestamp"
        return df

    def _fix_wind_speed_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fixes negative wind speed values using efficient, vectorized operations."""
        self.logger.log("Fixing negative wind speed values.", tag="fix")  # Use cleaned column names
        for col in self.anomaly_cols:
            if col in df.columns:
                # Replace values less than 0 with NaN, then forward-fill
                df[col] = df[col].mask(df[col] < 0).ffill().bfill()
        return df

    def _impute_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values in the DataFrame using the configured strategy.

        Args:
            df (pd.DataFrame): The DataFrame with missing values to impute.

        Returns:
            pd.DataFrame: The imputed DataFrame with no missing values.
        """
        self.logger.log(f"Starting imputation with method: '{self.imputation_method}'", tag=["impute", "info"])
        df_imputed = df.copy()
        numeric_cols = df_imputed.select_dtypes(include=np.number).columns.tolist()

        if self.imputation_method == "seasonal":
            for ts in df_imputed[df_imputed[numeric_cols].isnull().any(axis=1)].index:
                ref_ts = ts - pd.DateOffset(years=1)
                if ref_ts in df_imputed.index and df_imputed.loc[ref_ts].notnull().all():
                    df_imputed.loc[ts, numeric_cols] = df_imputed.loc[ref_ts, numeric_cols].values
            # Fallback for any remaining NaNs (e.g., first year)
            df_imputed.ffill(inplace=True)
            df_imputed.bfill(inplace=True)
        elif self.imputation_method == "forward_fill":
            df_imputed.ffill(inplace=True)
            df_imputed.bfill(inplace=True)  # bfill for any leading NaNs
        elif self.imputation_method == "rolling_mean":
            df_imputed.fillna(df_imputed.rolling(window=3, min_periods=1, center=True).mean(), inplace=True)
        elif self.imputation_method == "rolling_median":
            df_imputed.fillna(df_imputed.rolling(window=3, min_periods=1, center=True).median(), inplace=True)
        else:
            raise InvalidParametersError(InvalidParametersError.IMPUTATION_METHODS)

        return df_imputed

    def _resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resamples the DataFrame to a different time frequency.

        Args:
            df (pd.DataFrame): The DataFrame to resample.

        Returns:
            pd.DataFrame: The resampled DataFrame.
        """
        self.logger.log(f"Starting resampling to '{self.sampling_rate}'", tag=["resample", "info"])
        return df.resample(self.sampling_rate).agg(self.resample_method)

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters the DataFrame to include only the years within the configured range.

        Args:
            df (pd.DataFrame): The DataFrame to filter.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        self.logger.log(f"Applying year filter for range: {self.filter_range}", tag=["filter", "info"])

        if not isinstance(df.index, pd.DatetimeIndex):
            raise InvalidParametersError(InvalidParametersError.INDEX_DTYPE)
        return df[(df.index.year >= self.filter_range[0]) & (df.index.year <= self.filter_range[1])]

        # --- NEW: Integrated Sinusoidal Encoder Method ---

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies sinusoidal encoding to the DataFrame's DatetimeIndex and cyclic columns.
        """
        self.logger.log("Applying sinusoidal encoding.", tag="encode")

        df_encoded = df.copy()
        try:
            if not isinstance(df_encoded.index, pd.DatetimeIndex):
                timestamps = pd.to_datetime(df_encoded.index)

                # --- 1. Timestamp Features (from the index) ---
                self.logger.log("Encoding timestamp features.", tag="encode")

                # Hour of day (0-23)
                df_encoded["hour_sin"] = np.sin(2 * np.pi * timestamps.hour / 24.0)
                df_encoded["hour_cos"] = np.cos(2 * np.pi * timestamps.hour / 24.0)

                # Day of month (handles different month lengths correctly)
                days_in_month = timestamps.days_in_month
                df_encoded["day_of_month_sin"] = np.sin(2 * np.pi * timestamps.day / days_in_month)
                df_encoded["day_of_month_cos"] = np.cos(2 * np.pi * timestamps.day / days_in_month)

                # Day of year (handles leap years correctly)
                days_in_year = np.where(timestamps.is_leap_year, 366, 365)
                df_encoded["day_of_year_sin"] = np.sin(2 * np.pi * timestamps.dayofyear / days_in_year)
                df_encoded["day_of_year_cos"] = np.cos(2 * np.pi * timestamps.dayofyear / days_in_year)

                # Month of year (1-12)
                df_encoded["month_sin"] = np.sin(2 * np.pi * timestamps.month / 12.0)
                df_encoded["month_cos"] = np.cos(2 * np.pi * timestamps.month / 12.0)

                # --- 2. Other Cyclical Column Features ---
                if self.cyclic_cols:
                    self.logger.log(f"Encoding cyclical columns: {self.cyclic_cols}", tag="encode")
                    for col in self.cyclic_cols:
                        if col in df_encoded.columns:
                            # Assuming the cycle is 360 (like degrees)
                            df_encoded[f"{col}_sin"] = np.sin(2 * np.pi * df_encoded[col] / 360.0)
                            df_encoded[f"{col}_cos"] = np.cos(2 * np.pi * df_encoded[col] / 360.0)
                        else:
                            self.logger.log(
                                f"Warning: Cyclical column '{col}' not found in DataFrame.",
                                tag="encode",
                                level="warning",
                            )
        except Exception as e:
            self.logger.log(f"Error: {e}", tag="encode", level="error")
            raise SinusoidalEncodingError(SinusoidalEncodingError.DATETIME_TYPE_ERROR) from e
        else:
            return df_encoded

    def run(self, save_output: bool = True) -> pd.DataFrame:
        """Executes the full ETL pipeline based on the provided flags.

        This method enforces a strict operational order. It first loads the data,
        then validates the user's request. If transformations are requested, it
        mandates imputation. The order is always Impute -> Resample -> Filter.
        Finally, it saves the output to a fixed location for DVC.

        Args:
            save (bool): Whether to save the processed DataFrame to a CSV file.
        Returns:
            pd.DataFrame: The final, processed DataFrame with a reset index.

        Raises:
            DataProcessingError: If `resample` or `filter_years` is True but
                `impute` is False, or if any other processing step fails.
        """
        self.logger.log("ETL pipeline started.", tag=["etl", "info"])

        try:
            df = self._extract_and_load()
            if self.fix_data_errors:
                df = self._fix_wind_speed_errors(df)
            if self.impute:
                df = self._impute_data(df)
            if self.encode_sinusoidal:
                df = self._encode_features(df)
            if self.resample:
                df = self._resample_data(df)
            if self.filter_years:
                df = self._filter_data(df)

            self.processed_df = df

            if save_output:
                self._save_output()

            self.logger.log("Preprocessing complete. ETL pipeline finished successfully.", tag=["etl", "info"])
            return self.processed_df.reset_index()
        except Exception as e:
            self.logger.log(f"ETL pipeline failed: {e}", tag=["etl", "error"], level="warning")
            raise DataProcessingError(DataProcessingError.EXTRACTION_FAILED) from e

    def _save_output(self):
        """Saves the final processed DataFrame to a fixed path for DVC tracking."""
        if self.processed_df is None:
            # --- FIX for TRY003 ---
            raise DataProcessingError(DataProcessingError.NO_DATA_TO_SAVE)
            # --- END OF FIX ---

        self.logger.log(f"Saving pre-processed data to {self.PREPROCESSED_CSV_PATH}", tag=["io", "info"])
        self.processed_df.reset_index().to_csv(self.PREPROCESSED_CSV_PATH, index=False)
        self.logger.log("Save complete.", tag=["io", "info"])
