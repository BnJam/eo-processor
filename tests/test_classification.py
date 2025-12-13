import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from eo_processor import random_forest_predict
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
