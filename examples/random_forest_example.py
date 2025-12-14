import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from eo_processor import random_forest_predict

# Add the parent directory to the Python path to allow importing from `tests`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tests.utils import sklearn_to_json

def main():
    """Main function to run the random forest example."""
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
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

    accuracy = np.mean(predictions == sklearn_predictions)
    print(f"Accuracy compared to scikit-learn: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
