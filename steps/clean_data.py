import logging
from typing_extensions import Annotated
from typing import Tuple
from src.data_cleaning import DataPreprocessingStrategy, DataSplitStrategy
from src.data_cleaning import DataCleaning

import pandas as pd
from zenml import step


@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
]:

    try:
        preprocessing_strategy = DataPreprocessingStrategy()
        cleaner = DataCleaning(df, strategy=preprocessing_strategy)
        clean_df = cleaner.handle_data()

        # Splitting strategy
        split_strategy = DataSplitStrategy()
        splitter = DataCleaning(clean_df, strategy=split_strategy)
        X_train, X_test, y_train, y_test = splitter.handle_data()
        logging.info("Data loading and preprocessing completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(
            "An error occurred while loading or preprocessing {}".format(e)
            )
        raise e
