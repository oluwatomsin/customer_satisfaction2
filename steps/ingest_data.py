import logging

import pandas as pd
from zenml import step


class IngestData:
    """Ingesting data from a specified file path
    """
    def __init__(self, file_path: str):
        """This instantiated the path to file that will be read
        by the get_data function

        Args:
            file_path (str): The path to file.
        """
        self.file_path = file_path

    def get_data(self) -> pd.DataFrame:
        """This reads the data as a pandas DataFrame

        Returns:
            pd.DataFrame: The file as a DataFrame.
        """
        logging.info(f"Loading data from {self.file_path}")
        return pd.read_csv(self.file_path)


@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """Ingesting data from data path.

    Args:
        data_path (str): The path to the file to be r

    Returns:
        pd.DataFrame: Returns the loaded dataset as a pandas dataframe.
    """
    try:
        ingest = IngestData(file_path=data_path)
        data = ingest.get_data()
        return data
    except Exception as e:
        logging.info(f"An error occurred while ingesting data: {e}")
        raise e 
