"""Main script to run the ETL pipeline for the Jena Climate dataset.
This script uses argparse to gather configuration for the DataETL class,
instantiates the class, and runs the pipeline.
It allows users to specify parameters such as imputation method, resampling method,
sampling rate, and whether to filter the data by year range.

# src/components/data_etl.py

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

    IMPUTATION_METHODS = "Valid imputation values: ['seasonal', 'forward_fill', 'rolling_mean', 'rolling_median']"
    RESAMPLE_METHODS = "Valid resampling values: ['mean', 'median', 'first', 'last']"
    SAMPLING_RATES = "Valid sampling values: ['1H', '1D', '1M', '1Y']"
    FILTER_YEARS = "Valid filter years: ['True', 'False']"
    FILTER_RANGE = "Valid years range, inclusive: [2009, 2016]"
    FILTER_COUNT = "Valid filter count: 2 bounds"
    FILTER_BOUNDS = "First bound must be less than second bound"


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

    The class also provides a logger for tracking observations with optional metadata.
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
        run(self, impute: bool = False, resample: bool = False, save: bool = False):
            Executes the ETL pipeline based on the provided flags.
    """

    # --- Configuration Parameters ---
    imputation_method: Literal["seasonal", "forward_fill", "rolling_mean", "rolling_median"] = "seasonal"
    resample_method: Literal["mean", "median", "first", "last"] = "mean"
    sampling_rate: Literal["1H", "1D", "1Y"] = "1H"
    filter_years: bool = False
    filter_range: list[int] = field(default_factory=lambda: [2014, 2017])

    # --- Internal State ---
    raw_data_exists: bool = field(init=False, repr=False)
    processed_df: Optional[pd.DataFrame] = field(default=None, repr=False, init=False)

    # --- Class Constants ---
    RAW_DATA_DIR: ClassVar[str] = "./data/raw"
    TRANSFORMED_DATA_DIR: ClassVar[str] = "./data/transformed"
    RAW_CSV_PATH: ClassVar[str] = os.path.join(RAW_DATA_DIR, "jena_climate_2009_2016.csv")
    TRANSFORMED_CSV_PATH: ClassVar[str] = os.path.join(TRANSFORMED_DATA_DIR, "data.csv")

    def __post_init__(self):
        """Initializes the logger and ensures data directories exist."""
        self.logger = ObservationLogger(log_file="data_etl.log")
        self.logger.update_tags({
            "etl": "Main ETL Process Control",
            "extract": "Data Extraction Step",
            "load": "Data Loading & Cleaning Step",
            "impute": "Imputation Step",
            "filter": "Filtering Step",
            "resample": "Resampling Step",
            "error": "Error Messages",
            "validation": "Input validation",
            "io": "File Input/Output",
        })
        # The check happens here, when the object is created.
        self.raw_data_exists = os.path.exists(self.RAW_CSV_PATH)
        os.makedirs(self.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(self.TRANSFORMED_DATA_DIR, exist_ok=True)
        print(f"DataETL initialized. Raw data exists: {self.raw_data_exists}")

        # --- NEW VALIDATION LOGIC ---
        self.logger.log("Validating configuration parameters.", tag=["validation", "info"])
        # Check if filter_range has exactly two values
        if self.filter_years:
            if len(self.filter_range) != 2:
                error_msg = f"{InvalidParametersError.FILTER_COUNT}, got {self.filter_range}"
                self.logger.log(error_msg, tag=["validation", "error"], level="warning")
                raise InvalidParametersError()
            else:
                start_year, end_year = self.filter_range
                # Check if the years are within the allowed bounds (2009-2016)
                if not (2009 <= start_year <= 2016 and 2009 <= end_year <= 2016):
                    error_msg = f"{InvalidParametersError.FILTER_COUNT}, got {self.filter_range}"
                    self.logger.log(error_msg, tag=["validation", "error"], level="warning")
                    raise InvalidParametersError(error_msg)
                # Check if the start year is before the end year
                if start_year > end_year:
                    error_msg = f"{InvalidParametersError.FILTER_BOUNDS}, got {self.filter_range}"
                    self.logger.log(error_msg, tag=["validation", "error"], level="warning")
                    raise InvalidParametersError(error_msg)
            self.logger.log("Configuration parameters are valid.", tag=["validation", "info"])
        # --- END OF NEW VALIDATION LOGIC ---

    def _extract_and_load(self) -> pd.DataFrame:
        """Extracts the dataset and performs initial cleaning and preparation.

        This method handles downloading the zipped data, extracting the CSV,
        standardizing column names, parsing the datetime column, and reindexing
        to a consistent 10-minute frequency, which introduces NaNs for DVC to track.

        Returns:
            pd.DataFrame: The raw, cleaned DataFrame with a DatetimeIndex.

        Raises:
            DataProcessingError: If the data cannot be downloaded, extracted, or read.
        """
        if not self.raw_data_exists:
            self.logger.log("Raw data not found. Starting download.", tag=["extract", "info"])
            TEMP_CACHE_DIR = "./temp_cache"

            try:
                # 1. Create a temporary directory for the download
                os.makedirs(TEMP_CACHE_DIR, exist_ok=True)

                # 2. Download the zip file to our controlled temporary location
                zip_path = keras.utils.get_file(
                    origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
                    fname="jena_climate.zip",
                    cache_dir=os.path.abspath(TEMP_CACHE_DIR),
                    extract=False,  # We will handle extraction ourselves
                )

                # 3. Unpack the archive to the same temporary location
                shutil.unpack_archive(zip_path, TEMP_CACHE_DIR)

                # 4. Move the final CSV file from the temp location to its permanent home
                extracted_csv_path = os.path.join(TEMP_CACHE_DIR, "jena_climate_2009_2016.csv")
                shutil.move(extracted_csv_path, self.RAW_CSV_PATH)
                self.logger.log(f"Raw data successfully saved to {self.RAW_CSV_PATH}", tag=["extract", "info"])
                self.raw_data_exists = True  # Update state

            except Exception as e:
                raise DataProcessingError(DataProcessingError.EXTRACTION_FAILED) from e
            finally:
                # 5. Clean up the temporary directory completely
                if os.path.exists(TEMP_CACHE_DIR):
                    shutil.rmtree(TEMP_CACHE_DIR)
        else:
            self.logger.log("Raw data already exists. Skipping download.", tag=["extract", "info"])
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
        return df[(df.index.year >= self.filter_range[0]) & (df.index.year <= self.filter_range[1])]

    def run(self, impute: bool = False, resample: bool = False, save: bool = True) -> pd.DataFrame:
        """Executes the full ETL pipeline based on the provided flags.

        This method enforces a strict operational order. It first loads the data,
        then validates the user's request. If transformations are requested, it
        mandates imputation. The order is always Impute -> Resample -> Filter.
        Finally, it saves the output to a fixed location for DVC.

        Args:
            impute (bool): If True, performs data imputation.
            resample (bool): If True, performs data resampling.

        Returns:
            pd.DataFrame: The final, processed DataFrame with a reset index.

        Raises:
            DataProcessingError: If `resample` or `filter_years` is True but
                `impute` is False, or if any other processing step fails.
        """
        self.logger.log("ETL pipeline started.", tag=["etl", "info"])

        if (resample or self.filter_years) and not impute:
            error_msg = "Invalid request: Imputation is mandatory for resampling or filtering. Set impute=True."
            self.logger.log(error_msg, tag=["validation", "error"], level="warning")
            raise DataProcessingError(error_msg)

        try:
            self.processed_df = self._extract_and_load()
            if impute:
                self.processed_df = self._impute_data(self.processed_df)
            if resample:
                self.processed_df = self._resample_data(self.processed_df)
            if self.filter_years:
                self.processed_df = self._filter_data(self.processed_df)

            if save:
                self.save_output()

            self.logger.log("ETL pipeline finished successfully.", tag=["etl", "info"])
            return self.processed_df.reset_index()
        except Exception as e:
            self.logger.log(f"ETL pipeline failed: {e}", tag=["etl", "error"], level="warning")
            raise

    def save_output(self):
        """Saves the final processed DataFrame to a fixed path for DVC tracking."""
        if self.processed_df is None:
            # --- FIX for TRY003 ---
            raise DataProcessingError(DataProcessingError.NO_DATA_TO_SAVE)
            # --- END OF FIX ---

        self.logger.log(f"Saving transformed data to {self.TRANSFORMED_CSV_PATH}", tag=["io", "info"])
        self.processed_df.reset_index().to_csv(self.TRANSFORMED_CSV_PATH, index=False)
        self.logger.log("Save complete.", tag=["io", "info"])
