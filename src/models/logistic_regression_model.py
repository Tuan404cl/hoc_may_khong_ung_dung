"""Independent Module 4: Logistic Regression Gesture Classifier."""

from sklearn.linear_model import LogisticRegression
from src.models.base_evaluator import BaseGestureModelModule


class LogisticRegressionModelModule(BaseGestureModelModule):
    model_key = "lr"
    model_display_name = "Logistic Regression (Multinomial L2)"

    def build_estimator(self):
        return LogisticRegression(C=1.5, max_iter=1000, random_state=42)


if __name__ == "__main__":
    module = LogisticRegressionModelModule()
    module.train_and_evaluate()
