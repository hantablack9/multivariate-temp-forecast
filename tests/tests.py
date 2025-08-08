"""Unit test for the DataETL class to ensure it can load data correctly.
This test checks if the DataETL class can successfully load the raw data
and return a DataFrame with the expected structure.

This test is part of the data processing pipeline and is designed to
validate the functionality of the DataETL class.

It is expected to be run in an environment where the DataETL class is
correctly implemented and the necessary data files are available.

The test will fail if the DataETL class cannot load the data or if the
resulting DataFrame does not meet the expected criteria.

This test is crucial for ensuring the integrity of the data processing
pipeline and should be run regularly as part of the development process.

"""

import pandas as pd
from src.components.data_etl import DataETL  # Note the correct import path


def test_etl_loads_data():
    """
    Tests if the DataETL class can successfully load the raw data.
    """
    # Arrange
    etl_processor = DataETL()

    # Act
    df = etl_processor.run(impute=False, resample=False)

    # Assert
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'timestamp' in df.columns
