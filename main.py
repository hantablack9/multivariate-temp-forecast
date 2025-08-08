# main.py

"""
Main script to run the ETL pipeline for the Jena Climate dataset.
This script uses argparse to gather configuration for the DataETL class,
instantiates the class, and runs the pipeline.
It allows users to specify parameters such as imputation method, resampling method,
sampling rate, and whether to filter the data by year range.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse

from src.components.data_etl import DataETL, DataProcessingError


def main():
    """
    Main function to run the ETL pipeline from the command line.

    This script uses argparse to gather configuration for the DataETL class,
    instantiates the class, and runs the pipeline.
    """
    # 1. --- Argument Parser Setup ---
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run the ETL pipeline for the Jena Climate dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Shows default values in help
    )

    # --- Arguments for the DataETL class configuration ---
    parser.add_argument(
        "--imputation-method",
        type=str,
        default="seasonal",
        choices=["seasonal", "forward_fill", "rolling_mean", "rolling_median"],
        help="Strategy to use for filling missing values.",
    )
    parser.add_argument(
        "--resample-method",
        type=str,
        default="mean",
        choices=["mean", "median", "first", "last"],
        help="Aggregation method for resampling.",
    )
    parser.add_argument(
        "--sampling-rate", type=str, default="1H", choices=["1H", "1D", "1Y"], help="Target frequency for resampling."
    )
    parser.add_argument(
        "--filter-years",
        action="store_true",  # Makes this a flag; default is False
        help="Enable this flag to filter the data by year range.",
    )
    parser.add_argument(
        "--filter-range",
        type=int,
        nargs=2,  # Expects two integer values
        default=[2014, 2017],
        metavar=("START_YEAR", "END_YEAR"),
        help="The start and end years for filtering (inclusive start, exclusive end).",
    )

    # --- Arguments for the run() method ---
    parser.add_argument(
        "--impute",
        action="store_true",
        help="Enable this flag to perform data imputation. Mandatory for resampling or filtering.",
    )
    parser.add_argument("--resample", action="store_true", help="Enable this flag to perform data resampling.")

    # 2. --- Parse the arguments ---
    args = parser.parse_args()

    # 3. --- Run the Pipeline ---
    try:
        print("--- Running Data Processing Pipeline with Command-Line Arguments ---")

        # Instantiate the processor with arguments from the parser
        etl_processor = DataETL(
            imputation_method=args.imputation_method,
            resample_method=args.resample_method,
            sampling_rate=args.sampling_rate,
            filter_years=args.filter_years,
            filter_range=args.filter_range,
        )

        print("\nConfiguration:")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        print("-" * 30)

        # Run the pipeline with run-time flags
        etl_processor.run(impute=args.impute, resample=args.resample)

        print("\nPipeline finished successfully.")
        print(f"Transformed data is available at: {etl_processor.TRANSFORMED_CSV_PATH}")

    except DataProcessingError as e:
        print(f"\nPIPELINE FAILED: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
