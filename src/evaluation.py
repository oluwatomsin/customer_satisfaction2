from abc import ABC, abstractmethod
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score


class Evaluation(ABC):
    """An abstract class for the model evaluation
    """

    @abstractmethod
    def calculate_scores(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray
    ):
        ...


class AccuracyScore(Evaluation):

    def calculate_scores(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray
    ):
        try:
            logging.info("Calculation Accuracy")
            score = accuracy_score(y_true=y_true, y_pred=y_pred)
            logging.info("The model accuracy: {}".format(score))
            return score
        except Exception as e:
            logging.error("Error while evaluating accuracy: {}".format(e))
            raise e


class PrecisionScore(Evaluation):

    def calculate_scores(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray
    ):
        try:
            logging.info("Calculation Precision")
            score = precision_score(y_true=y_true, y_pred=y_pred)
            logging.info("The model precision: {}".format(score))
            return score
        except Exception as e:
            logging.error("Error while evaluating precision: {}".format(e))
            raise e


class FScore(Evaluation):

    def calculate_scores(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray
    ):
        try:
            logging.info("Calculation f1 score")
            score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info("The model f1 score: {}".format(score))
            return score
        except Exception as e:
            logging.error("Error while evaluating f1 score: {}".format(e))
            raise
