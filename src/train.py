"""
File to contain all model training functions for ICU data.
"""
from typing import List, Optional
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from tqdm import tqdm

from evaluation import get_metrics_and_curves, crossvalidated_curve_stats, confidence_intervals


def run_lr_train(X_train, X_test, y_train, y_test, tpr_match=None, fpr_match=None, verbose=True):
    """
    Fit logistic regression with L2 regularization.
    Z-score scaling prior to model fit.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if verbose:
        print("Training logistic model with l2 penalty")

    logisticReg = LogisticRegression(penalty="l2", max_iter=500, fit_intercept=True)
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

    fold_metrics, fold_curves = get_metrics_and_curves(
        y_test=y_test, y_pred=y_pred, y_probs=lr_probs, tpr_match=tpr_match, fpr_match=fpr_match
    )

    result_dict = {
        "model": logisticReg,
        "test_pred": predictions,
        "metrics": fold_metrics,
        "curves": fold_curves,
    }

    return result_dict


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

        results = run_lr_train(X_train, X_test, y_train, y_test, verbose=verbose)

        cv_results[i] = results
        metric_list.append(results["metrics"])

        # Update progress bar
        pbar.update(1)
        mean_roc = np.around(np.mean([x['AUC ROC'] for x in metric_list]), 4)
        std_roc = np.around(np.std([x['AUC ROC'] for x in metric_list]), 4)
        mean_pr = np.around(np.mean([x['AUC PR'] for x in metric_list]), 4)
        std_pr = np.around(np.std([x['AUC PR'] for x in metric_list]), 4)
        pbar.set_description(
            f"DEWS2 {folds}FOLDCV | AUROC {mean_roc} (+/- {std_roc})" + \
            f" | AUPRC {mean_pr} (+/- {std_pr})",
            refresh=True
        )
    pbar.close()

    cv_results["CV_AVG"] = {
        "metrics": pd.DataFrame(metric_list).mean().to_dict(),
        "curves": crossvalidated_curve_stats(cv_results)
    }

    return cv_results


def train_logistic_model_bootstrapped(
        icu_df: pd.DataFrame, independent_vars: List[str], dependant_var: str, n_bootstraps: int,
        tpr_match: Optional[float] = None, fpr_match: Optional[float] = None,
        verbose: bool = False, test_icu_df=pd.DataFrame()
):
    """
    Train logistic regression model with given independent variables to classify dependant var.
    Bootstrap and test on OUT OF BAG samples if test_icu_df is not provided.
    Test on test_icu_df if provided.
    """
    metric_list = []
    cv_results = {}
    completed_bootstraps = 0
    pbar = tqdm(total=n_bootstraps, desc="DEWS2 BOOTSTRAP", leave=True)
    while completed_bootstraps < n_bootstraps:

        #print(completed_bootstraps, "bootstraps")

        # bootstrap by sampling with replacement
        sample_class0 = patient_sample(icu_df[icu_df[dependant_var] == 0])
        sample_class1 = patient_sample(icu_df[icu_df[dependant_var] == 1])
        sample = pd.concat([sample_class0, sample_class1])

        if test_icu_df.empty or test_icu_df.equals(icu_df):
            oob = icu_df.iloc[list(set(icu_df.index) - set(sample.index))]
        else:
            sample_class0 = patient_sample(test_icu_df[test_icu_df[dependant_var] == 0])
            sample_class1 = patient_sample(test_icu_df[test_icu_df[dependant_var] == 1])
            oob = pd.concat([sample_class0, sample_class1])

        results = run_lr_train(sample[independent_vars], oob[independent_vars], sample[dependant_var], oob[dependant_var], verbose=verbose)

        cv_results[completed_bootstraps] = results
        metric_list.append(results["metrics"])
        completed_bootstraps += 1

        # Update progress bar
        pbar.update(1)
        mean_roc, lower_roc, upper_roc = confidence_intervals([x['AUC ROC'] for x in metric_list])
        mean_pr, lower_pr, upper_pr = confidence_intervals([x['AUC PR'] for x in metric_list])
        pbar.set_description(
            f"DEWS2 BOOTSTRAP | AUROC {mean_roc} (CI {lower_roc} - {upper_roc})" +\
            f" | AUPRC {mean_pr} (CI {lower_pr} - {upper_pr})",
            refresh=True
        )
    pbar.close()

    cv_results["CV_AVG"] = {
        "metrics": pd.DataFrame(metric_list).mean().to_dict(),
        "CI_95_upper": pd.DataFrame(metric_list).apply(lambda x: np.sort(np.array(x))[int(0.975*len(x))]).to_dict(),
        "CI_95_lower": pd.DataFrame(metric_list).apply(lambda x: np.sort(np.array(x))[int(0.025*len(x))]).to_dict(),
        "curves": crossvalidated_curve_stats(cv_results)
    }

    return cv_results


def patient_sample(icu_df, replace=True, frac=1):
    """ Take a sample of patient observations."""
    unique_ids = icu_df["ADMISSION_ID"].unique()
    sampled_ids = np.random.choice(unique_ids, size=frac*len(unique_ids), replace=replace)
    return icu_df[icu_df["ADMISSION_ID"].isin(sampled_ids)].copy(deep=True)


def train_logistic_model_CV_grouped(
        icu_df: pd.DataFrame, independent_vars: List[str], dependant_var: str, groups: pd.Series,
        folds: int = 5, tpr_match: Optional[float] = None, fpr_match: Optional[float] = None,
        verbose: bool = False
):
    """
    Train logistic regression model with given independent variables to classify dependant var.
    Stratified K fold crossvalidated with non-overlapping patient groups.
    """
    X = icu_df[independent_vars]
    y = icu_df[dependant_var]

    cv_results = {}
    skf = StratifiedGroupKFold(n_splits=folds)
    skf.get_n_splits(X, y, groups)
    metric_list = []
    pbar = tqdm(total=folds, desc=f"DEWS2 {folds}FOLD CV", leave=True)
    for i, indices in enumerate(skf.split(X, y, groups)):
        train_index, test_index = indices
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if verbose:
            print(f"FOLD: {i}")

        results = run_lr_train(X_train, X_test, y_train, y_test, verbose=verbose)

        cv_results[i] = results
        metric_list.append(results["metrics"])

        # Update progress bar
        pbar.update(1)
        mean_roc = np.around(np.mean([x['AUC ROC'] for x in metric_list]), 4)
        std_roc = np.around(np.std([x['AUC ROC'] for x in metric_list]), 4)
        mean_pr = np.around(np.mean([x['AUC PR'] for x in metric_list]), 4)
        std_pr = np.around(np.std([x['AUC PR'] for x in metric_list]), 4)
        pbar.set_description(
            f"DEWS2 {folds}FOLDCV | AUROC {mean_roc} (+/- {std_roc})" + \
            f" | AUPRC {mean_pr} (+/- {std_pr})",
            refresh=True
        )
    pbar.close()

    cv_results["CV_AVG"] = {
        "metrics": pd.DataFrame(metric_list).mean().to_dict(),
        "curves": crossvalidated_curve_stats(cv_results)
        }

    return cv_results
