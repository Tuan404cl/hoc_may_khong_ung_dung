"""Independent Module 3: Gaussian Naive Bayes Gesture Classifier."""

from sklearn.naive_bayes import GaussianNB
from src.models.base_evaluator import BaseGestureModelModule


class NaiveBayesModelModule(BaseGestureModelModule):
    model_key = "nb"
    model_display_name = "Gaussian Naive Bayes (GNB)"

    def build_estimator(self):
        return GaussianNB(var_smoothing=0.08)


if __name__ == "__main__":
    module = NaiveBayesModelModule()
    module.train_and_evaluate()
