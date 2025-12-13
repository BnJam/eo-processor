# Complex Workflows for EO Analysis

This document outlines two complex workflows that demonstrate the power and flexibility of the `eo-processor` library. These workflows leverage the high-performance Rust UDFs to perform sophisticated Earth Observation analyses that would be slow and memory-intensive in pure Python.

## Workflow 1: Deforestation Monitoring with Automated Change Detection

This workflow automates the detection of deforestation events by combining temporal analysis, spectral indices, and spatial filtering. It is designed to be robust to seasonal changes and sensor noise, providing a reliable and scalable solution for monitoring forest cover.

### Steps

1. **Temporal Compositing:**
   - A time series of Sentinel-2 images is composited using the `temporal_composite` function with the `median` method. This creates a cloud-free, seasonally representative baseline image.
   - The same is done for a more recent time series to create a "current" image.

2. **Spectral Index Calculation:**
   - The Normalized Difference Vegetation Index (NDVI) is calculated for both the baseline and current images using the `ndvi` function. This highlights changes in vegetation cover.

3. **Change Detection:**
   - The difference in NDVI between the baseline and current images is calculated to create a delta-NDVI image. This highlights areas where vegetation has been lost or gained.

4. **Texture Analysis:**
   - The `texture_entropy` function is applied to the delta-NDVI image. This helps to differentiate between actual deforestation and other types of change, such as agricultural harvesting, which tend to have a more uniform texture.

5. **Morphological Filtering:**
   - A `binary_opening` operation is applied to the entropy-filtered image to remove small, isolated pixels that are likely to be noise. This is followed by a `binary_closing` operation to fill in small gaps in the detected deforestation areas.

6. **Zonal Statistics:**
   - The `zonal_stats` function is used to calculate the total area of deforestation within predefined administrative boundaries or protected areas.

## Workflow 2: Crop Yield Prediction with Multi-Sensor Data Fusion

This workflow combines data from multiple sensors to predict crop yields. It leverages the library's ability to perform complex classifications and temporal analysis to create a robust and accurate prediction model.

### Steps

1. **Data Fusion:**
   - Time series data from Sentinel-2 and Landsat 8 are combined into a single, harmonized time series. This involves resampling the data to a common spatial resolution and performing radiometric normalization.

2. **Complex Classification:**
   - The `complex_classification` function is used to identify different crop types within the study area. This is a critical step, as different crops have different growth cycles and yield potentials.

3. **Time-Series Analysis:**
   - For each crop type, a time series of NDVI is extracted. The `detect_breakpoints` function is used to identify key phenological stages, such as green-up, peak growth, and senescence.

4. **Feature Engineering:**
   - A set of features is engineered from the NDVI time series, including the maximum NDVI, the length of the growing season, and the rate of green-up.
   - The `texture_entropy` of the NDVI at peak growth is also calculated as a proxy for crop health and uniformity.

5. **Yield Prediction:**
   - A machine learning model, such as a random forest or gradient boosting model, is trained to predict crop yields based on the engineered features. The model is trained on historical yield data and validated against a held-out test set.

6. **Zonal Statistics:**
   - The `zonal_stats` function is used to aggregate the predicted yields to the field or administrative boundary level.
