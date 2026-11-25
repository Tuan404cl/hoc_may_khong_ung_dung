import numpy as np
import pytest

from src.config import DATASET_CSV_PATH, POSE_CLASSES
from src.features import get_feature_names
from src.models.knn_model import KNNModelModule
from src.models.decision_tree_model import DecisionTreeModelModule
from src.models.naive_bayes_model import NaiveBayesModelModule
from src.models.logistic_regression_model import LogisticRegressionModelModule
from src.models.svm_model import SVMModelModule


@pytest.mark.parametrize(
    "module_cls,expected_key",
    [
        (KNNModelModule, "knn"),
        (DecisionTreeModelModule, "dt"),
        (NaiveBayesModelModule, "nb"),
        (LogisticRegressionModelModule, "lr"),
        (SVMModelModule, "svm"),
    ],
)
def test_independent_model_module_train_evaluate_and_predict(module_cls, expected_key, tmp_path):
    mod = module_cls(model_dir=tmp_path / "models", results_dir=tmp_path / "results")
    assert mod.model_key == expected_key

    metrics = mod.train_and_evaluate(csv_path=DATASET_CSV_PATH)
    assert "accuracy" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics
    min_acc = 0.85 if expected_key == "nb" else 0.94
    assert metrics["accuracy"] >= min_acc
    assert len(metrics["confusion_matrix"]) == len(POSE_CLASSES)

    # Check saved artifacts for this independent module
    assert (tmp_path / "models" / f"{expected_key}_model.pkl").exists()
    assert (tmp_path / "results" / expected_key / "metrics.json").exists()
    assert (tmp_path / "results" / expected_key / "confusion_matrix.png").exists()
    assert (tmp_path / "results" / expected_key / "error_analysis.md").exists()

    # Check independent inference
    dummy_features = np.zeros(len(get_feature_names()), dtype=np.float64)
    assert len(dummy_features) == 54
    pred_label, confidence = mod.predict_pose(dummy_features)
    assert pred_label in POSE_CLASSES
    assert 0.0 <= confidence <= 1.0
