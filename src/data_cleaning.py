import logging
from abc import ABC, abstractmethod
from typing_extensions import Tuple, Annotated
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[
            pd.DataFrame,
            pd.Series
            ]:
        ...


class DataPreprocessingStrategy(DataStrategy):
    """This implements the strategy for preprocessing our data

    Args:
        DataStrategy (class): Inherited base strategy class
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data['satisfaction'] = data['satisfaction'].map({
                "satisfied": 1,
                "dissatisfied": 0
            })
            data["Gender"] = data["Gender"].map({
                "Female": 0,
                "Male": 1
            })
            data["Customer Type"] = data["Customer Type"].map({
                "Loyal Customer": 1,
                "disloyal Customer": 0
            })
            data["Type of Travel"] = data["Type of Travel"].map({
                "Personal Travel": 0,
                "Business travel": 1
            })
            data["Class"] = data["Class"].map({
                "Eco": 0,
                "Eco Plus": 1,
                "Business": 2
            })

            data = data.dropna()
            return data
        except Exception as e:
            logging.error("Error preprocessing the data: {}".format(e))
            raise e


class DataSplitStrategy(DataStrategy):
    """This inherits from the strategy class to create a
    class for splitting the dataset

    Args:
        DataStrategy (class): Inherited base strategy class
    """
    def handle_data(self, data: pd.DataFrame) -> Tuple[
        Annotated[pd.DataFrame, "X_train"],
        Annotated[pd.DataFrame, "X_test"],
        Annotated[pd.Series, "y_train"],
        Annotated[pd.Series, "y_test"]
    ]:
        """This function splits the data into train and test sets for the
        dependent and independent variables.

        Args:
            data (pd.DataFrame): The input dataframe to be splitted

        Returns:
            Union[pd.Series, pd.DataFrame]: The train and test datasets
        """
        try:
            X = data.drop("satisfaction", axis=1)
            y = data['satisfaction']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=True, random_state=100
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error while splitting data {}".format(e))
            raise e


class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handling the data based on the specified strategy.
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error handling data {}".format(e))
            raise e
