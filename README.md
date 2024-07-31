# Predictive modelling pipeline for death or ICU admission in eOBS

## Overview

This project involves the processing and modeling of ICU data to predict death or ICU admission based on various physiological measurements and other time series based features. The pipeline involves multiple steps including data extraction, transformation, feature engineering, and model training to achieve reliable predictions.

An API deployment of the models trained here can be found at https://github.com/ATayls/DEWS_fastapi

## Table of Contents

- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
  - [Extract and Transform](#extract-and-transform)
  - [Feature Engineering](#feature-engineering)
  - [Model Training](#model-training)
- [Configuration](#configuration)
- [Usage](#usage)
- [Results](#results)
- [Plots](#plots)
- [Export](#export)
- [Files and Directories](#files-and-directories)

## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- sklearn
- tsfresh
- shap
- tqdm
- openpyxl


## Data Pipeline

### Extract and Transform

The ETL (Extract, Transform, Load) process handles loading, preprocessing, and feature engineering of the data. The `ETL` function:

1. Loads the data from the specified filename.
2. Applies preprocessing steps.
3. Creates additional features.
4. Saves or loads the processed dataset.

### Feature Engineering

The feature engineering process involves creating time series features, calculating rolling averages, standard deviations, and slopes. It includes functions like:

- `create_time_delta`: Creates a time variable.
- `create_diff`: Calculates differences from previous values.
- `create_rolling`: Calculates rolling averages and standard deviations.
- `create_expanding`: Calculates expanding averages and standard deviations.
- `create_ts_base_features`: Combines multiple time series features.
- `create_slopes_cached`: Calculates slopes for variables.

### Model Training

The model training process involves training logistic regression models using cross-validation and bootstrapping techniques. It includes:

- `run_lr_train`: Trains a logistic regression model.
- `train_logistic_model_cv`: Performs cross-validation.
- `train_logistic_model_bootstrapped`: Uses bootstrapping for model training.
- `train_logistic_model_CV_grouped`: Cross-validation with non-overlapping patient groups.

## Configuration

Configuration settings are handled in the `settings.py` file, including directory paths for data, processed data, saved results, plots, and models.

## Usage

To run the main experiment, execute the `run.py` script:

```bash
python run.py
```

This script will perform the following steps:

1. Load and preprocess the training and testing data.
2. Perform feature engineering on the data.
3. Train logistic regression models using both cross-validation and bootstrapping.
4. Evaluate the models on the test set.
5. Save the results, models, and plots.

## Results

The results of the model training and evaluation are saved in CSV format in the `SAVED_RESULTS_DIR` directory. The results include metrics such as AUROC and AUPRC along with confidence intervals.

## Plots

The script generates several plots to visualize the model performance and feature importance:

- ROC and PR curves for cross-validation and test sets.
- Permutation importance plots.
- SHAP value summaries.

These plots are saved in the `PLOTS_DIR` directory.

## Export

The processed data, model predictions, and metrics can be exported to Excel files for further analysis. This is handled by the `export_as_excel.py` module.

## Files and Directories

- `run.py`: Main script to run the experiment.
- `preprocessing.py`: Contains preprocessing functions.
- `feature_engineering.py`: Contains feature engineering functions.
- `train.py`: Contains model training functions.
- `settings.py`: Configuration settings.
- `plots.py`: Functions to generate plots.
- `export_as_excel.py`: Functions to export results to Excel.

- `evaluation.py`: Utilities around model evaluation.
- `utils.py`: General Utilities.
