import logging
from sklearn.base import ClassifierMixin
import pandas as pd

import mlflow
from zenml import step
from zenml.client import Client

from .config import ModelNameConfig
from src.model_dev import LogisticRegressionModel


experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
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
            mlflow.sklearn.autolog()
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
