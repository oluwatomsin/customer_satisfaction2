import logging
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig

import pandas as pd
from zenml import step
from src.model_dev import LogisticRegressionModel


@step
def train_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: ModelNameConfig) -> ClassifierMixin:
    """Model training step.

    Args:
        X_train (pd.DataFrame): The training data[indep variable]
        y_train (pd.DataFrame): The training data[dep variable]
    """
    try:
        model = None
        if config.model_name == "RandomForestClassifier":
            model = LogisticRegressionModel().train(
                X_train=X_train,
                y_train=y_train)
            return model
        else:
            logging.error("Model name not recognized")
            raise ValueError("Model not supported")
    except Exception as e:
        logging.error("An error occurred while training {}".format(e))
        raise e
