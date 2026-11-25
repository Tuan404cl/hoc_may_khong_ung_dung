"""Independent Module 1: K-Nearest Neighbors (KNN) Gesture Classifier."""

from sklearn.neighbors import KNeighborsClassifier
from src.models.base_evaluator import BaseGestureModelModule


class KNNModelModule(BaseGestureModelModule):
    model_key = "knn"
    model_display_name = "K-Nearest Neighbors (KNN, k=5)"

    def build_estimator(self):
        return KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski", p=2)


if __name__ == "__main__":
    module = KNNModelModule()
    module.train_and_evaluate()
