"""Independent Module 5: Support Vector Machine (SVM) Gesture Classifier."""

from sklearn.svm import SVC
from src.models.base_evaluator import BaseGestureModelModule


class SVMModelModule(BaseGestureModelModule):
    model_key = "svm"
    model_display_name = "Support Vector Machine (RBF Kernel)"

    def build_estimator(self):
        return SVC(kernel="rbf", C=3.0, gamma="scale", random_state=42)


if __name__ == "__main__":
    module = SVMModelModule()
    module.train_and_evaluate()
