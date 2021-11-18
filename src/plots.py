from typing import Optional, List, Any
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.metrics import auc
import shap


def roc_curve(tpr: np.array, fpr: np.array, name: str,
              tpr_std: Optional[np.array] = None,
              show_chance: Optional[bool] = True,
              area_under_curve: Optional[float] = None,
              figure: Optional[plt.figure] = None,
              **kwargs
):
    if figure is None:
        figure = plt.figure()
    if not area_under_curve:
        area_under_curve = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{name} (AUC=%0.3f)' % (area_under_curve), **kwargs)

    if tpr_std is not None:
        tprs_upper = np.minimum(tpr + tpr_std, 1)
        tprs_lower = np.maximum(tpr - tpr_std, 0)
        plt.fill_between(fpr, tprs_lower, tprs_upper, color='grey', alpha=.15)

    if show_chance:
        plt.plot([0, 1], [0, 1], color='grey', alpha=.25, linestyle='dotted')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    return figure


def pr_curve(precisions: np.array, recalls: np.array, name: str,
             precisions_std: Optional[np.array] = None,
             area_under_curve: Optional[float] = None,
             figure: Optional[plt.figure] = None,
             **kwargs
):
    if figure is None:
        figure = plt.figure()
    if not area_under_curve:
        area_under_curve = auc(recalls, precisions)

    plt.plot(recalls, precisions, label=f'{name} (AUC=%0.3f)' % (area_under_curve), **kwargs)

    if precisions_std is not None:
        precisions_upper = np.minimum(precisions + precisions_std, 1)
        precisions_lower = np.maximum(precisions - precisions_std, 0)
        plt.fill_between(recalls, precisions_lower, precisions_upper, color='grey', alpha=.15)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="upper right")
    plt.title('Precision Recall Curve')

    return figure


def compare_cv_results(news2_cv_results: dict, dews_cv_results: dict, display_std: bool = False):
    ### ROC ###
    news2_fpr = news2_cv_results["CV_AVG"]["curves"]["ROC"]["FPR"]
    news2_tpr = news2_cv_results["CV_AVG"]["curves"]["ROC"]["TPR"]
    news2_tpr_std = news2_cv_results["CV_AVG"]["curves"]["ROC"]["TPR STD"]
    news2_auroc = news2_cv_results["CV_AVG"]["metrics"]["AUC ROC"]
    dews_fpr = dews_cv_results["CV_AVG"]["curves"]["ROC"]["FPR"]
    dews_tpr = dews_cv_results["CV_AVG"]["curves"]["ROC"]["TPR"]
    dews_tpr_std = dews_cv_results["CV_AVG"]["curves"]["ROC"]["TPR STD"]
    dews_auroc = dews_cv_results["CV_AVG"]["metrics"]["AUC ROC"]

    if display_std:
        figure = roc_curve(news2_tpr, news2_fpr, "NEWS-2", news2_tpr_std, area_under_curve=news2_auroc,
                           linestyle='dashdot')
        figure = roc_curve(dews_tpr, dews_fpr, "DEWS", dews_tpr_std, area_under_curve=dews_auroc,
                           figure=figure)
    else:
        figure = roc_curve(news2_tpr, news2_fpr, "NEWS-2", area_under_curve=news2_auroc, linestyle='dashdot')
        figure = roc_curve(dews_tpr, dews_fpr, "DEWS", area_under_curve=dews_auroc, figure=figure)
    plt.show()

    ## PR CURVE ##
    news2_recall = news2_cv_results["CV_AVG"]["curves"]["PR"]["R"]
    news2_precision = news2_cv_results["CV_AVG"]["curves"]["PR"]["P"]
    news2_precision_std = news2_cv_results["CV_AVG"]["curves"]["PR"]["P STD"]
    news2_auprc = news2_cv_results["CV_AVG"]["metrics"]["AUC PR"]
    dews_recall = dews_cv_results["CV_AVG"]["curves"]["PR"]["R"]
    dews_precision = dews_cv_results["CV_AVG"]["curves"]["PR"]["P"]
    dews_precision_std = dews_cv_results["CV_AVG"]["curves"]["PR"]["P STD"]
    dews_auprc = dews_cv_results["CV_AVG"]["metrics"]["AUC PR"]

    if display_std:
        figure = pr_curve(news2_precision, news2_recall, "NEWS-2", news2_precision_std, area_under_curve=news2_auprc,
                          linestyle='dashdot')
        figure = pr_curve(dews_precision, dews_recall, "DEWS", dews_precision_std, area_under_curve=dews_auprc,
                          figure=figure)
    else:
        figure = pr_curve(news2_precision, news2_recall, "NEWS-2", area_under_curve=news2_auprc, linestyle='dashdot')
        figure = pr_curve(dews_precision, dews_recall, "DEWS", area_under_curve=dews_auprc, figure=figure)
    plt.show()

def shap_linear_summary(model: Any, data_scaled: DataFrame, feature_names: List[str]):
    """ SHAP summary for Linear model """
    print("Calculating SHAP values")
    explainer = shap.LinearExplainer(model, data_scaled)
    shap_values = explainer.shap_values(data_scaled)
    shap.summary_plot(shap_values, data_scaled, feature_names=feature_names)


def permutation_importance_plot(cv_results: dict, feature_names: List[str], title: Optional[str] = None,
                                save_path: Optional[Path] = None):
    # Feature importance
    feature_importances = DataFrame(
        [list(v["model"].coef_[0]) for k, v in cv_results.items() if k != "CV_AVG"],
        columns=feature_names
    )
    if save_path:
        feature_importances.to_csv(save_path, index=False)
    feature_importances = feature_importances.reindex(
        feature_importances.mean().sort_values().index, axis=1
    )
    feature_importances.boxplot(rot=90, fontsize=6)
    if title:
        plt.title(title)
    plt.show()