"""Registry of the 5 independent Machine Learning gesture recognition modules."""

from src.models.knn_model import KNNModelModule
from src.models.decision_tree_model import DecisionTreeModelModule
from src.models.naive_bayes_model import NaiveBayesModelModule
from src.models.logistic_regression_model import LogisticRegressionModelModule
from src.models.svm_model import SVMModelModule

MODEL_REGISTRY = {
    "knn": KNNModelModule,
    "dt": DecisionTreeModelModule,
    "nb": NaiveBayesModelModule,
    "lr": LogisticRegressionModelModule,
    "svm": SVMModelModule,
}


def get_model_module(model_key: str):
    key = model_key.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key]()
