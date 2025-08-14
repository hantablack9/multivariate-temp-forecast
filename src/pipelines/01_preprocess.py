"""
Preprocessing stage

This script orchestrates the data extraction, imputation, resampling, and filtering stages.
It loads the configuration from the YAML file, instantiates the DataETL class, and runs the pipeline.

dependencies: src/multivariate_temp_forecast/data_etl.py
outs: data/processed/preprocessed_data.csv

"""

import os
import sys

import yaml

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

"""
# ModuleNotFoundError: No module named 'src' when running DVC repro
--- THIS IS THE FIX ---
Add the project root directory to the Python path
This allows us to use absolute imports starting from 'src'
os.path.abspath(__file__) gives the absolute path of the current script
os.path.dirname() gets the directory of the script (e.g., .../src/pipelines)
# We go up two levels to get to the project root
"""
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
# --- END OF FIX ---

from src.multivariate_temp_forecast.data_etl import DataETL

# 1. Load the configuration from the YAML file
with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# 2. Instantiate your class, passing the loaded parameters.
# The ** operator unpacks the dictionary, overriding the defaults in your class.
etl_processor = DataETL(**params["preprocess"])

# 3. Run the pipeline


etl_processor.run(save_output=True)

print("Preprocessing stage complete.")
