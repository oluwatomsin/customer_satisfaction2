from abc import ABC, abstractmethod
import logging
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin


class Model(ABC):
    """ Abstract class for all training model"""

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """This trains the machine learning model

        Args:
            X_train (pd.DataFrame): The training data[indep variable]
            y_train (pd.DataFrame): The training data[dep variable]
        """
        ...


class LogisticRegressionModel(Model):
    """Training a logistic regression model
    """
    def train(
            self,
            X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
        """Trains Logistic Model

        Args:
            X_train (pd.DataFrame): The training data[indep variable]
            y_train (pd.DataFrame): The training data[dep variable]
        """
        try:
            reg = RandomForestClassifier()
            reg.fit(X_train, y_train)
            logging.info("Training of model completed")
            return reg
        except Exception as e:
            logging.error("Error while training model {}".format(e))
            raise e
