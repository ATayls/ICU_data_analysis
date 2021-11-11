"""
File to contain model evaluation functions.
"""
from typing import Optional
from copy import copy
from sklearn.metrics import (average_precision_score, roc_auc_score, roc_curve, confusion_matrix,
                             precision_recall_curve, auc)
import numpy as np


def get_metrics_and_curves(
        y_test: np.array, y_pred: np.array, y_probs: np.array,
        tpr_match: Optional[float]=None, fpr_match: Optional[float]=None
):
    """
    Calculate performance metrics and ROC PR curves given y predictions and truth.
    :param y_test: truth
    :param y_pred: classification predictions
    :param y_probs: classification probabilities
    :param tpr_match: true positive ratio to match.
    :param fpr_match: false positive ratio to match.
    :return: Metric and curve dictionaries.
    """

    # Assuming two class problem, take positive class
    y_probs = y_probs[:, 1]

    # calculate scores
    logit_roc_auc = roc_auc_score(y_test, y_probs)

    # calculate roc curves
    fpr, tpr, roc_thresh = roc_curve(y_test, y_probs)

    # calculate PR curves
    precision, recall, pr_thresh = precision_recall_curve(y_test, y_probs)

    # Average precision score
    avg_p = average_precision_score(y_test, y_probs)

    # calculate scores
    logit_pr_auc = auc(recall, precision)

    if tpr_match and fpr_match:
        raise ValueError("Can not match fpr and tpr simultaneously")
    if tpr_match:
        y_pred = matched_thresh_y_pred(y_probs, tpr, roc_thresh, tpr_match)
    elif fpr_match:
        y_pred = matched_thresh_y_pred(y_probs, fpr, roc_thresh, fpr_match)

    # Confusion matrix
    CM = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = CM[0][0], CM[0][1], CM[1][0], CM[1][1]

    metrics = {
        "accuracy": (TP + TN) / (TP + TN + FP + FN),
        "precision": TP / (TP + FP),
        "sensitivity": TP / (TP + FN),
        "specificity": TN / (TN + FP),
        "F1": TP / (TP + (FP + FN) / 2),
        "AUC ROC": logit_roc_auc,
        "AUC PR": logit_pr_auc,
        "PR AvgP": avg_p
    }

    curves = {
        "PR": {"P": precision, "R": recall, "thr": pr_thresh},
        "ROC": {"TPR": tpr, "FPR": fpr, "thr": roc_thresh}
    }

    return metrics, curves


def confidence_intervals(scores: np.array, interval: float = 0.95, decimals: int = 4):
    """
    Computing the lower and upper bound of the confidence interval.
    """
    sorted_scores = copy(np.array(scores))
    sorted_scores.sort()

    mean_score = sorted_scores.mean()
    if len(scores) > 2:
        thresh = (1-interval)/2
        confidence_lower= sorted_scores[
            int(thresh * len(sorted_scores))
        ]
        confidence_upper= sorted_scores[
            int((1-thresh) * len(sorted_scores))
        ]
    else:
        confidence_lower = mean_score
        confidence_upper = mean_score

    mean_score_r = np.around(mean_score, decimals)
    confidence_lower_r = np.around(confidence_lower, decimals)
    confidence_upper_r = np.around(confidence_upper, decimals)
    return mean_score_r, confidence_lower_r, confidence_upper_r


def crossvalidated_curve_stats(cv_results):
    """
    Calculate the average and std of multiple Precision-recall and ROC curves.
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    :param cv_results: list of crossvalidation results.
    :return:
    """
    uniform_xaxis_values = np.linspace(0, 1, 100)
    tprs, precisions = [], []
    for fold_i, fold_outputs in cv_results.items():
        # Intepolate curve y values along uniform linspace
        interp_tpr = np.interp(
            uniform_xaxis_values,
            fold_outputs["curves"]["ROC"]["FPR"],
            fold_outputs["curves"]["ROC"]["TPR"]
        )
        interp_precision = np.interp(
            uniform_xaxis_values,
            fold_outputs["curves"]["PR"]["R"][::-1],
            fold_outputs["curves"]["PR"]["P"][::-1]
        )
        interp_tpr[-1] = 1.0
        interp_precision[-1] = 0.0

        tprs.append(interp_tpr)
        precisions.append(interp_precision)

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_precisions = np.mean(precisions, axis=0)
    std_precisions = np.std(precisions, axis=0)

    average_curve_dict = {
        "PR": {"P": mean_precisions, "R": uniform_xaxis_values, "P STD": std_precisions},
        "ROC": {"TPR": mean_tpr, "FPR": uniform_xaxis_values, "TPR STD": std_tpr}
    }

    return average_curve_dict


def matched_thresh_y_pred(y_probs: np.array, tpr: np.array, roc_thresh: np.array, tpr_match: float):
    """ Recalculate y_pred at a threshold that matches input tpr"""
    for i, tpr_val in enumerate(tpr):
        if tpr_val>tpr_match:
            break
    matched_threshold = roc_thresh[i-1]+(
            (tpr_match-tpr[i-1])*((roc_thresh[i]-roc_thresh[i-1])/(tpr[i]-tpr[i-1]))
    )
    y_pred = (y_probs > matched_threshold).astype(int)
    return y_pred
