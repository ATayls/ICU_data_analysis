"""
File to contain all model training functions for ICU data.
"""
from typing import List, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from tqdm import tqdm

from evaluation import get_metrics_and_curves, crossvalidated_curve_stats


def run_lr_train(X_train, X_test, y_train, y_test, verbose=True):
    """
    Fit logistic regression with L2 regularization.
    Z-score scaling prior to model fit.
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param verbose:
    :return:
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if verbose:
        print("Training logistic model with l2 penalty")

    logisticReg = LogisticRegression(penalty="l2", max_iter=500, fit_intercept=False)
    logisticReg.fit(X_train, y_train)

    # LR predicitions
    y_pred = logisticReg.predict(X_test)

    # LR probabilities
    lr_probs = logisticReg.predict_proba(X_test)

    if isinstance(y_test, pd.Series):
        y_test = np.array(y_test)

    predictions = {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_probs": lr_probs,
    }

    return logisticReg, predictions


def train_logistic_model_cv(
        icu_df: pd.DataFrame, independent_vars: List[str], dependant_var: str,
        folds: int = 5, tpr_match: Optional[float] = None, fpr_match: Optional[float] = None,
        verbose: bool = False
):
    """
    Train logistic regression model with given independent variables to classify dependant var.
    Stratified K fold crossvalidated.
    """
    X = icu_df[independent_vars].copy()
    y = icu_df[dependant_var].copy()

    cv_results = {}
    skf = StratifiedKFold(n_splits=folds)
    skf.get_n_splits(X, y)
    metric_list = []
    pbar = tqdm(total=folds, desc=f"DEWS2 {folds}FOLD CV", leave=True)
    for i, indices in enumerate(skf.split(X, y)):
        train_index, test_index = indices
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if verbose:
            print(f"FOLD: {i}")

        logisticReg, test_pred = run_lr_train(X_train, X_test, y_train, y_test, verbose=verbose)

        fold_metrics, fold_curves = get_metrics_and_curves(**test_pred, tpr_match=tpr_match,
                                                           fpr_match=fpr_match)
        cv_results[i] = {
            "model": logisticReg,
            "test_pred": test_pred,
            "metrics": fold_metrics,
            "curves": fold_curves,
        }
        metric_list.append(fold_metrics)
        current_mean = np.around(np.mean([x['AUC ROC'] for x in metric_list]),4)
        current_std = np.around(np.std([x['AUC ROC'] for x in metric_list]),4)
        pbar.update(1)
        pbar.set_description(
            f"DEWS2 {folds}FOLD CV {current_mean} (+/- {current_std})",
            refresh=True
        )
    pbar.close()

    cv_results["CV_AVG"] = {
        "metrics": pd.DataFrame(metric_list).mean().to_dict(),
        "curves": crossvalidated_curve_stats(cv_results)
    }

    return cv_results
