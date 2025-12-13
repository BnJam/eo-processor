use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    root: DecisionNode,
}

impl DecisionTree {
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut current_node = &self.root;
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

#[derive(Serialize, Deserialize, Debug)]
pub struct RandomForest {
    trees: Vec<DecisionTree>,
}

impl RandomForest {
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

use numpy::{PyReadonlyArray2, PyArray1};
use pyo3::prelude::*;

#[pyfunction]
pub fn random_forest_predict<'py>(
    py: Python<'py>,
    model_json: &str,
    features: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    let forest: RandomForest = serde_json::from_str(model_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to deserialize model: {}", e)))?;

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
