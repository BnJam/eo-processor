import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from eo_processor import random_forest_predict, random_forest_train
from .utils import sklearn_to_json


def test_random_forest_predict():
    """Test the random_forest_predict function."""
    # Generate synthetic data
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        random_state=42,
        shuffle=False,
    )

    # Train a scikit-learn RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    # Convert the trained model to JSON
    model_json = sklearn_to_json(clf)

    # Perform inference using the Rust implementation
    predictions = random_forest_predict(model_json, X)

    # Compare with scikit-learn's predictions
    sklearn_predictions = clf.predict(X)

    assert np.array_equal(predictions, sklearn_predictions)


def test_random_forest_train_and_predict():
    """Test the full train-and-predict cycle."""
    # Generate synthetic data
    X, y = make_classification(
        n_samples=200,
        n_features=15,
        n_informative=10,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        shuffle=True,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Convert labels to float64
    y_train = y_train.astype(np.float64)

    # Train the model using the Rust implementation
    model_json = random_forest_train(
        X_train,
        y_train,
        n_estimators=100,
        min_samples_split=3,
        max_depth=10,
    )

    # Perform inference
    predictions = random_forest_predict(model_json, X_test)

    # Check accuracy
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy >= 0.75, (
        f"Accuracy of {accuracy:.2f} is below the threshold of 0.75"
    )


def test_random_forest_train_rejects_label_length_mismatch():
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    y = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="labels length"):
        random_forest_train(x, y, n_estimators=5, min_samples_split=2, max_depth=3)


def test_random_forest_train_rejects_non_finite_features():
    x = np.array([[0.1, np.nan], [0.3, 0.4]], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        random_forest_train(x, y, n_estimators=5, min_samples_split=2, max_depth=3)


def test_random_forest_predict_rejects_feature_width_mismatch():
    x_train = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float64)
    y_train = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    model_json = random_forest_train(
        x_train, y_train, n_estimators=5, min_samples_split=2, max_depth=3
    )

    x_bad = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    with pytest.raises(ValueError, match="column count"):
        random_forest_predict(model_json, x_bad)
