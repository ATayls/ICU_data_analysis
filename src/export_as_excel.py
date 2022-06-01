from typing import List, Dict
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def export(
    df: pd.DataFrame,
    trained_lr_model: LogisticRegression,
    original_fields: List[str],
    feature_list: List[str],
    fitted_scaler: StandardScaler,
    metrics: Dict[str, int],
    save_path: Path,
):
    print("---Exporting full model to excel")
    # Scale with fitted scaler
    feature_mean = fitted_scaler.mean_
    feature_std = np.sqrt(fitted_scaler.var_)
    scaling_metrics = pd.DataFrame([feature_mean, feature_std], index=["Mean", "Std"],
                                   columns=feature_list)
    scaled_features = fitted_scaler.transform(df[feature_list])

    # Apply Logisitic model
    lr = pd.DataFrame(scaled_features, columns=feature_list).dot(trained_lr_model.coef_[0])
    lr = lr + trained_lr_model.intercept_[0]
    assert(trained_lr_model.intercept_scaling == 1)
    probs = 1 - lr.apply(lambda x: 1 / (1 + np.exp(x)))
    probs_df = pd.concat([df[["ADMISSION_ID", "OBS_DAYS_SINCE_ADMISSION", "OBS_TIME"]], probs], axis=1)
    probs_df.columns = ["ADMISSION_ID", "OBS_DAYS_SINCE_ADMISSION", "OBS_TIME", "PROBABILITY"]

    f_coefs = pd.concat([
        pd.DataFrame(trained_lr_model.intercept_, columns=["INTERCEPT"]),
        pd.DataFrame(trained_lr_model.coef_, columns=feature_list)
    ],axis=1)

    # Write to excel
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(save_path) as writer:
        df[original_fields].to_excel(writer, sheet_name='Original data', index=False)
        df[feature_list].to_excel(writer, sheet_name='Engineered features', index=False)
        scaling_metrics.to_excel(writer, sheet_name='Scaling metrics')
        pd.DataFrame(scaled_features, columns=feature_list).to_excel(writer, sheet_name='Scaled features', index=False)
        f_coefs.to_excel(writer, sheet_name='Feature Coeffiecents', index=False)
        probs_df.to_excel(writer, sheet_name='LR probability', index=False)
        pd.DataFrame(pd.Series(metrics)).to_excel(writer, sheet_name='Metrics', index=True)
