import logging
from sklearn.base import ClassifierMixin
from src.evaluation import PrecisionScore, FScore, AccuracyScore

from typing import Tuple
import pandas as pd
from typing_extensions import Annotated
import numpy as np
from zenml import step


@step
def evaluate_model(
        model: ClassifierMixin,
        X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[
                    Annotated[float, "acc"],
                    Annotated[float, "pre"],
                    Annotated[float, "f1"]]:
    """Function to evaluate the performance of the model based
    on specific metrics

    Args:
        model (ClassifierMixin): Model to evaluate
        X_test (np.ndarray): The test data
        y_test (np.ndarray): The test data
    """

    try:
        y_pred = model.predict(X_test)
        acc = AccuracyScore().calculate_scores(y_pred=y_pred, y_true=y_test)

        pre = PrecisionScore().calculate_scores(y_pred=y_pred, y_true=y_test)

        f1 = FScore().calculate_scores(y_pred=y_pred, y_true=y_test)

        return acc, pre, f1
    except Exception as e:
        logging.error("Error while evaluating {}".format(e))
        raise e
