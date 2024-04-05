from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """This contains the model configurations"""

    model_name: str = "LogisticRegression"
