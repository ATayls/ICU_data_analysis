"""
File to contain any NEWS2 specific functions.
"""
from typing import Optional
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
import numpy as np
from tqdm import tqdm

from evaluation import get_metrics_and_curves, confidence_intervals


def predictions_news2(
        icu_df: DataFrame,
        dependant_var: str,
        threshold: Optional[int] = None,
):
    """
    Get predictions using NEWS2 score and an applied threshold.
    :param icu_df: Dataframe of observation data
    :param threshold: NEWS2 classification threshold
    :param dependant_var: Target variable
    :return:
    """
    scaler = MinMaxScaler()
    scaler.fit(icu_df[["NEWS_2"]].values)

    if threshold:
        scaled_threshold = scaler.transform(np.array(threshold).reshape(-1, 1))
        scaled_threshold = np.asscalar(scaled_threshold)
    else:
        scaled_threshold = 0.5

    y_test = icu_df[dependant_var].values
    y_probs = scaler.transform(icu_df[["NEWS_2"]].values)[:, 0]
    y_pred = (y_probs > scaled_threshold).astype(int)
    y_probs = np.array([1 - y_probs, y_probs]).T
    return y_test, y_pred, y_probs


def bootstrap_news2(
        icu_df: DataFrame, n_bootstraps: int, dependant_var: str, threshold: Optional[int] = None,
):
    """
    Run the news2 predictions with n bootstraps.
    :param threshold:
    :param icu_df:
    :param n_bootstraps:
    :param dependant_var:
    :return:
    """
    metric_list = []
    completed_bootstraps = 0
    pbar = tqdm(total=n_bootstraps, desc="NEWS2 BOOTSTRAP", leave=True)
    while completed_bootstraps < n_bootstraps:
        # bootstrap by sampling with replacement
        sample = icu_df.sample(frac=1, replace=True).copy(deep=True)

        y_test, y_pred, y_probs = predictions_news2(
            sample, threshold=threshold, dependant_var=dependant_var
        )
        metrics, _ = get_metrics_and_curves(
            y_test, y_pred, y_probs
        )
        metric_list.append(metrics)
        completed_bootstraps += 1
        pbar.update(1)
        mean_auc, lower, upper = confidence_intervals([x['AUC ROC'] for x in metric_list])
        pbar.set_description(f"NEWS2 BOOTSTRAP {mean_auc} (CI {lower} - {upper})", refresh=True)
    pbar.close()

    return metric_list
