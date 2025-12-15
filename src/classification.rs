use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

type SplitData = (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>);

#[derive(Serialize, Deserialize, Debug)]
pub enum DecisionNode {
    Leaf {
        class_prediction: f64,
    },
    Node {
        feature_index: usize,
        threshold: f64,
        left: Box<DecisionNode>,
        right: Box<DecisionNode>,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DecisionTree {
    root: Option<DecisionNode>,
    max_depth: Option<i32>,
    min_samples_split: i32,
}

impl DecisionTree {
    pub fn new(max_depth: Option<i32>, min_samples_split: i32) -> Self {
        DecisionTree {
            root: None,
            max_depth,
            min_samples_split,
        }
    }

    pub fn fit(
        &mut self,
        features: &[Vec<f64>],
        labels: &[f64],
        n_features_to_consider: usize,
    ) {
        self.root = Some(self.build_tree(features, labels, 0, n_features_to_consider));
    }

    fn build_tree(
        &self,
        features: &[Vec<f64>],
        labels: &[f64],
        depth: i32,
        n_features_to_consider: usize,
    ) -> DecisionNode {
        if let Some(max_depth) = self.max_depth {
            if depth >= max_depth {
                return DecisionNode::Leaf {
                    class_prediction: self.calculate_leaf_value(labels),
                };
            }
        }

        if labels.len() < self.min_samples_split as usize {
            return DecisionNode::Leaf {
                class_prediction: self.calculate_leaf_value(labels),
            };
        }

        if labels.iter().all(|&l| l == labels[0]) {
            return DecisionNode::Leaf {
                class_prediction: labels[0],
            };
        }

        let best_split = self.find_best_split(features, labels, n_features_to_consider);

        if let Some((feature_index, threshold)) = best_split {
            let (left_features, left_labels, right_features, right_labels) =
                self.split_data(features, labels, feature_index, threshold);

            if left_labels.is_empty() || right_labels.is_empty() {
                return DecisionNode::Leaf {
                    class_prediction: self.calculate_leaf_value(labels),
                };
            }

            let left_child = self.build_tree(
                &left_features,
                &left_labels,
                depth + 1,
                n_features_to_consider,
            );
            let right_child = self.build_tree(
                &right_features,
                &right_labels,
                depth + 1,
                n_features_to_consider,
            );

            DecisionNode::Node {
                feature_index,
                threshold,
                left: Box::new(left_child),
                right: Box::new(right_child),
            }
        } else {
            DecisionNode::Leaf {
                class_prediction: self.calculate_leaf_value(labels),
            }
        }
    }

    fn find_best_split(
        &self,
        features: &[Vec<f64>],
        labels: &[f64],
        n_features_to_consider: usize,
    ) -> Option<(usize, f64)> {
        let mut best_gain = -1.0;
        let mut best_split: Option<(usize, f64)> = None;
        let n_features = features[0].len();
        let current_gini = self.gini_impurity(labels);

        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        feature_indices.shuffle(&mut thread_rng());
        let feature_subset = &feature_indices[..n_features_to_consider.min(n_features)];

        for &feature_index in feature_subset {
            let mut unique_thresholds = features
                .iter()
                .map(|row| row[feature_index])
                .collect::<Vec<f64>>();
            unique_thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            unique_thresholds.dedup();

            for &threshold in &unique_thresholds {
                let (left_labels, right_labels) =
                    self.split_labels(labels, features, feature_index, threshold);

                if left_labels.is_empty() || right_labels.is_empty() {
                    continue;
                }

                let p_left = left_labels.len() as f64 / labels.len() as f64;
                let p_right = right_labels.len() as f64 / labels.len() as f64;
                let gain = current_gini
                    - p_left * self.gini_impurity(&left_labels)
                    - p_right * self.gini_impurity(&right_labels);

                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some((feature_index, threshold));
                }
            }
        }
        best_split
    }

    fn split_data(
        &self,
        features: &[Vec<f64>],
        labels: &[f64],
        feature_index: usize,
        threshold: f64,
    ) -> SplitData {
        let mut left_features = Vec::new();
        let mut left_labels = Vec::new();
        let mut right_features = Vec::new();
        let mut right_labels = Vec::new();

        for (i, row) in features.iter().enumerate() {
            if row[feature_index] <= threshold {
                left_features.push(row.clone());
                left_labels.push(labels[i]);
            } else {
                right_features.push(row.clone());
                right_labels.push(labels[i]);
            }
        }
        (left_features, left_labels, right_features, right_labels)
    }

    fn split_labels(
        &self,
        labels: &[f64],
        features: &[Vec<f64>],
        feature_index: usize,
        threshold: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut left_labels = Vec::new();
        let mut right_labels = Vec::new();
        for (i, label) in labels.iter().enumerate() {
            if features[i][feature_index] <= threshold {
                left_labels.push(*label);
            } else {
                right_labels.push(*label);
            }
        }
        (left_labels, right_labels)
    }

    fn gini_impurity(&self, labels: &[f64]) -> f64 {
        let mut counts = HashMap::new();
        for &label in labels {
            *counts.entry(label as i64).or_insert(0) += 1;
        }

        let mut impurity = 1.0;
        for &count in counts.values() {
            let prob = count as f64 / labels.len() as f64;
            impurity -= prob.powi(2);
        }
        impurity
    }

    fn calculate_leaf_value(&self, labels: &[f64]) -> f64 {
        let mut counts = HashMap::new();
        for &label in labels {
            *counts.entry(label as i64).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(val, _)| val as f64)
            .unwrap_or(0.0)
    }

    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut current_node = self.root.as_ref().expect("Tree is not trained yet.");
        loop {
            match current_node {
                DecisionNode::Leaf { class_prediction } => {
                    return *class_prediction;
                }
                DecisionNode::Node {
                    feature_index,
                    threshold,
                    left,
                    right,
                } => {
                    if features[*feature_index] <= *threshold {
                        current_node = left;
                    } else {
                        current_node = right;
                    }
                }
            }
        }
    }
}

use rand::Rng;

#[derive(Serialize, Deserialize, Debug)]
pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_estimators: i32,
    max_depth: Option<i32>,
    min_samples_split: i32,
    max_features: Option<usize>,
}

impl RandomForest {
    pub fn new(
        n_estimators: i32,
        max_depth: Option<i32>,
        min_samples_split: i32,
        max_features: Option<usize>,
    ) -> Self {
        RandomForest {
            trees: Vec::with_capacity(n_estimators as usize),
            n_estimators,
            max_depth,
            min_samples_split,
            max_features,
        }
    }

    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[f64]) {
        let n_samples = features.len();
        let n_features = features[0].len();
        let n_features_to_consider = self
            .max_features
            .unwrap_or_else(|| (n_features as f64).sqrt() as usize);

        self.trees = (0..self.n_estimators)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let sample_indices: Vec<usize> = (0..n_samples)
                    .map(|_| rng.gen_range(0..n_samples))
                    .collect();

                let bootstrapped_features: Vec<Vec<f64>> = sample_indices
                    .iter()
                    .map(|&i| features[i].clone())
                    .collect();
                let bootstrapped_labels: Vec<f64> =
                    sample_indices.iter().map(|&i| labels[i]).collect();

                let mut tree = DecisionTree::new(self.max_depth, self.min_samples_split);
                tree.fit(
                    &bootstrapped_features,
                    &bootstrapped_labels,
                    n_features_to_consider,
                );
                tree
            })
            .collect();
    }

    pub fn predict(&self, features: &[f64]) -> Option<f64> {
        if self.trees.is_empty() {
            return None;
        }

        let predictions: Vec<f64> = self
            .trees
            .par_iter()
            .map(|tree| tree.predict(features))
            .collect();

        let mut counts = HashMap::new();
        for prediction in predictions {
            *counts.entry(prediction as i64).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(val, _)| val as f64)
    }
}

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
pub fn random_forest_train(
    _py: Python,
    features: PyReadonlyArray2<f64>,
    labels: PyReadonlyArray1<f64>,
    n_estimators: i32,
    min_samples_split: i32,
    max_depth: Option<i32>,
    max_features: Option<usize>,
) -> PyResult<String> {
    let features_array = features.as_array();
    let labels_array = labels.as_array();

    let features_vec: Vec<Vec<f64>> = features_array
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    let labels_vec: Vec<f64> = labels_array.to_vec();

    let mut forest = RandomForest::new(n_estimators, max_depth, min_samples_split, max_features);
    forest.fit(&features_vec, &labels_vec);

    serde_json::to_string(&forest).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to serialize model: {}", e))
    })
}

#[pyfunction]
pub fn random_forest_predict<'py>(
    py: Python<'py>,
    model_json: &str,
    features: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    let forest: RandomForest = serde_json::from_str(model_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Failed to deserialize model: {}",
            e
        ))
    })?;

    let features_array = features.as_array();
    let n_samples = features_array.shape()[0];

    let predictions: Vec<f64> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let feature_row: Vec<f64> = features_array.row(i).iter().cloned().collect();
            forest.predict(&feature_row).unwrap_or(f64::NAN)
        })
        .collect();

    Ok(PyArray1::from_vec(py, predictions))
}
