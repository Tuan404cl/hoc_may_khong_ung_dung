"""Independent Module 2: Decision Tree Gesture Classifier."""

from sklearn.tree import DecisionTreeClassifier
from src.models.base_evaluator import BaseGestureModelModule


class DecisionTreeModelModule(BaseGestureModelModule):
    model_key = "dt"
    model_display_name = "Decision Tree (CART, max_depth=12)"

    def build_estimator(self):
        return DecisionTreeClassifier(criterion="gini", max_depth=12, min_samples_split=4, random_state=42)


if __name__ == "__main__":
    module = DecisionTreeModelModule()
    module.train_and_evaluate()
