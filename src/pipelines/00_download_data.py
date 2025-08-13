"""
Entry point for the data download DVC stage.
Instantiates the DataETL class and calls its download helper method.

# In src/pipelines/00_download_data.py
"""
import os
import sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import yaml


# 1. Load the configuration from the YAML file
with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.multivariate_temp_forecast.data_etl import DataETL

# 2. Instantiate your class, passing the loaded parameters.
# The ** operator unpacks the dictionary, overriding the defaults in your class.
etl_processor = DataETL()

# 3. Run the pipeline
etl_processor.download_if_needed()
print("Download stage complete.")