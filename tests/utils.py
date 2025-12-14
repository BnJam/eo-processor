import json
import numpy as np

def sklearn_to_json(model):
    """Converts a scikit-learn RandomForestClassifier to a JSON serializable format."""
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_

        def build_tree(node_id):
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Leaf node
                class_prediction = np.argmax(tree.value[node_id])
                return {"Leaf": {"class_prediction": float(class_prediction)}}
            else:
                # Decision node
                return {
                    "Node": {
                        "feature_index": int(tree.feature[node_id]),
                        "threshold": float(tree.threshold[node_id]),
                        "left": build_tree(tree.children_left[node_id]),
                        "right": build_tree(tree.children_right[node_id]),
                    }
                }

        trees.append({
            "root": build_tree(0),
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
        })

    return json.dumps({
        "trees": trees,
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "min_samples_split": model.min_samples_split,
        "max_features": model.max_features if isinstance(model.max_features, int) else None,
    })
