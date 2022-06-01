"""
File to contain all feature engineering functions for ICU data.
"""

import time
import functools
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh import extract_features

import settings


def selective_fillna_columns(df_in: pd.DataFrame, mode="mean") -> pd.DataFrame:
    """Apply fill na only to columns that require."""
    df = df_in.copy()
    for i in df.columns[df.isnull().any(axis=0)]:  # ---Applying Only on variables with NaN values
        if mode == "mean":
            df[i].fillna(df[i].mean(), inplace=True)
        elif mode == "median":
            df[i].fillna(df[i].median(), inplace=True)
        else:
            raise ValueError(f"Unrecognised mode: {mode}")
    return df


def split_non_monotonic(icu_df: pd.DataFrame, split_points: Dict[str, Tuple[int, int]]) -> pd.DataFrame:
    """
    Non-monotonic relationships with dependant variable split into dichotomous variables.

     "We allowed for non-symmetric effects of continuous predictors by breaking each physiological
     measurement into two variables reflecting positive (min (0, value-median value)) and
     negative (min (0, abs (value-median value)) deviations from the median value."
     [Zhu Y Resuscitation. 2020;157:176-184. doi:10.1016/j.resuscitation.2020.10.037]
    """
    icu_df = icu_df.copy()
    print(f"--Spliting into POS NEG variables: {list(split_points.keys())}")
    for variable, split in split_points.items():
        assert(variable in icu_df.columns), f"{variable} not found in icu_df"
        icu_df[f"{variable}_POS"] = icu_df[variable].apply(lambda x: max(0, x - split[1]))
        icu_df[f"{variable}_NEG"] = icu_df[variable].apply(lambda x: abs(min(0, x - split[0])))
    return icu_df


def create_time_delta(icu_df: pd.DataFrame, column_name: str = "TIME") -> pd.DataFrame:
    """
    Create TIME variable.
    Combining days since admission and observation time to create overall time since admission.
    """
    icu_df = icu_df.copy()
    print("--Calculating time delta")
    days_delta = pd.to_timedelta(icu_df["OBS_DAYS_SINCE_ADMISSION"], unit="day", errors="coerce")
    time_col = pd.to_datetime(icu_df["OBS_TIME"], format="%H:%M:%S", errors="coerce")
    icu_df[column_name] = time_col + days_delta
    return icu_df


def create_diff(icu_df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """
    Calculate difference from previous value on selected variables.
    """
    icu_df = icu_df.copy()
    print("--Calculating diff features")
    DIFF = (
        icu_df[['ADMISSION_ID'] + variables]
            .groupby('ADMISSION_ID')
            .transform(lambda s: s.diff(1))
            .add_suffix("_DIFF")
    )
    DIFF = selective_fillna_columns(DIFF, mode="mean")
    return pd.concat([icu_df, DIFF], axis=1)


def create_rolling(icu_df: pd.DataFrame, variables: List[str], periods: int) -> pd.DataFrame:
    """
    Calculate rolling average and standard deviation over N periods on selected variables.
    """
    icu_df = icu_df.copy()
    print(f"--Calculating rolling avg and std with period:{periods}")
    ROLAVG = (
        icu_df.groupby('ADMISSION_ID')[variables]
            .transform(lambda s: s.rolling(periods, min_periods=1).mean())
            .add_suffix("_ROLAVG")
    )
    ROLSTD = (
        icu_df.groupby('ADMISSION_ID')[variables]
            .transform(lambda s: s.rolling(periods, min_periods=1).std())
            .add_suffix("_ROLSTD")
    )
    ROLSTD = selective_fillna_columns(ROLSTD, mode="mean")
    return pd.concat([icu_df, ROLAVG, ROLSTD], axis=1)


def create_expanding(icu_df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """
    Calculate expanding average and standard deviation on selected variables.
    """
    icu_df = icu_df.copy()
    print("--Calculating expanding avg and std")
    ALLAVG = (
        icu_df.groupby('ADMISSION_ID')[variables]
              .transform(lambda s: s.expanding(min_periods=1).mean())
              .add_suffix("_ALLAVG")
    )
    ALLSTD = (
        icu_df.groupby('ADMISSION_ID')[variables]
            .transform(lambda s: s.expanding(min_periods=1).std())
            .add_suffix("_ALLSTD")
    )
    ALLSTD = selective_fillna_columns(ALLSTD, mode="mean")
    return pd.concat([icu_df, ALLAVG, ALLSTD], axis=1)


def create_ts_base_features(
        icu_df: pd.DataFrame, variables: List[str], periods: int = 3
) -> pd.DataFrame:
    """
    Create time series features for input variables.
    Returns Dataframe with features as additional columns with suffix.
        - Difference from previous value
        - Rolling Average (N periods)
        - Rolling Standard deviation (N periods)
        - Expanding Average
        - Expanding Standard deviation
    """
    print("Calculating standard time series features")
    return (
        icu_df.pipe(create_diff, variables)
              .pipe(create_rolling, variables, periods)
              .pipe(create_expanding, variables)
    )


def create_slopes_cached(icu_df: pd.DataFrame, variables: List[str], periods: int) -> pd.DataFrame:
    """
    Calculate variable slopes using a least squares polynomial fit over latest N periods.
    This Function uses NP poly fit and caches results for speed up.
    Slope calculation does not take the time delta between observations into account.
    """
    def calc_slope(x):
        slope = np.polyfit(range(len(x)), x, 1)[0]
        return slope

    @functools.lru_cache(maxsize=256 * 10)
    def memo_calc_slope(x1, x2, x3):
        x = [x1, x2, x3]
        return calc_slope(x)

    def hashable_calc_slope(x):
        x = x - x[0]
        return memo_calc_slope(x[0], x[1], x[2])

    icu_df = icu_df.copy()
    print(f"--Calculating rolling slope wth period:{periods}")
    ROLSLOPE = (
        icu_df.groupby('ADMISSION_ID')[variables]
              .transform(lambda s: s.rolling(periods, min_periods=3).apply(hashable_calc_slope))
              .add_suffix("_SLOPE")
    )
    return pd.concat([icu_df, ROLSLOPE], axis=1)


def create_ts_slope_features(
        icu_df: pd.DataFrame, variables: List[str], periods: int, time_col: str = "TIME",
        create_slope_categories: bool = True,
        stable_thresholds: Optional[Dict[str, Tuple[float, float]]] = None
) -> pd.DataFrame:
    """
    Calculate variable slopes both linear trend and linear trend timewise.
    Utilise tsfresh for speed up and timewise slopes.
    """
    icu_df = icu_df.copy()
    print("Calculating tsfresh time series features")
    original_columns = list(icu_df.columns.copy())

    # Time delta column must be present in the dataframe
    assert(time_col in icu_df.columns), f"{time_col} not found in dataframe"

    # Roll timeseries
    print("--Calculating ROLLED dataframe ", time.strftime("%H:%M:%S", time.localtime()))
    icu_df_rolled = roll_time_series(
        icu_df[["ADMISSION_ID", time_col] + variables].dropna(subset=[time_col]),
        column_id="ADMISSION_ID",
        column_sort=time_col,
        min_timeshift=2,
        max_timeshift=periods-1,
        n_jobs=0
    )

    # Extract slope features
    print("--Calculating slope features ", time.strftime("%H:%M:%S", time.localtime()))
    fc_parameters = {
        "linear_trend": [{'attr': 'slope'}],
        "linear_trend_timewise": [{'attr': 'slope'}],
    }
    icu_df_features = extract_features(
        icu_df_rolled[["id", time_col] + variables].set_index(time_col),
        column_id="id",
        default_fc_parameters=fc_parameters,
        n_jobs=0
    )

    # Rename output feature columns
    mapping = {"linear_trend": "_SLOPE", "linear_trend_timewise": "_SLOPE_TIMEWISE"}
    icu_df_features = icu_df_features.rename(
        columns={
            i: i.split("__")[0]+mapping[i.split("__")[1]] for i in icu_df_features.columns.to_list()
        }
    )

    # merge features back into original dataframe
    SLOPE_features = (
        icu_df_features.reset_index(drop=False)
                       .rename(columns={"level_0": "ADMISSION_ID", "level_1": time_col})
    )
    icu_df_ts = icu_df.merge(SLOPE_features, how="left", on=["ADMISSION_ID", time_col])

    new_slope_cols = list(set(icu_df_ts.columns) - set(original_columns))

    # Categorise slopes
    if create_slope_categories and stable_thresholds:
        icu_df_ts = categorise_slopes(icu_df_ts, stable_thresholds)
    elif create_slope_categories and not stable_thresholds:
        raise ValueError("Stable thresholds must be defined for slope categorisation")

    new_cat_cols = list(set(icu_df_ts.columns) - set(original_columns + new_slope_cols))

    # Fill nan slopes with 0 slope
    icu_df_ts[new_slope_cols] = selective_fillna_columns(icu_df_ts[new_slope_cols], mode="mean")
    icu_df_ts[new_cat_cols] = selective_fillna_columns(icu_df_ts[new_cat_cols], mode="mean")

    return icu_df_ts


def categorise_slopes(icu_df: pd.DataFrame,
                      stable_thresholds: Dict[str, Tuple[float, float]],
):
    """
    Categorise the SLOPE_TIMEWISE Variables.
    """
    icu_df = icu_df.copy()
    print(f"--Categorising slopes: {list(stable_thresholds.keys())}")
    for var, thresholds in stable_thresholds.items():
        icu_df[f"{var}_SLOPE_CAT"] = icu_df.apply(
            lambda row: categorise_obs_slope(row, var, thresholds), axis=1
        )
    return icu_df


def categorise_obs_slope(row: pd.Series, var_name: str, thresholds: Tuple[float, float]):
    """
    Slope categorisation to apply to pandas dataframe row-wise.
    """
    if np.isnan(row[var_name + "_SLOPE_TIMEWISE"]):
        return row[var_name + "_SLOPE_TIMEWISE"]

    stable_min, stable_max = thresholds
    stable_min /= 24
    stable_max /= 24
    if var_name == "O2_SATS":
        if row[var_name + "_SLOPE_TIMEWISE"] < stable_min:
            # Worsening
            return 4
        elif row[var_name + "_SLOPE_TIMEWISE"] > stable_max:
            # Improving
            return 2
        else:
            # Stable
            return 3
    elif ("INSPIRED_O2" in var_name) or ("INSP_O2" in var_name) :
        if row[var_name+"_SLOPE_TIMEWISE"] < stable_min:
            # Improving
            return 2
        elif row[var_name+"_SLOPE_TIMEWISE"] > stable_max:
            # Worsening
            return 4
        else:
            # Stable
            return 3
    elif row[var_name+"_POS"] > 0:
        if row[var_name+"_SLOPE_TIMEWISE"] < stable_min:
            # High risk improving
            return 2
        elif row[var_name+"_SLOPE_TIMEWISE"] > stable_max:
            # high risk worsening
            return 4
        else:
            # high risk stable
            return 3
    elif row[var_name+"_NEG"] > 0:
        if row[var_name + "_SLOPE_TIMEWISE"] < stable_min:
            # high risk and Worsening
            return 4
        elif row[var_name + "_SLOPE_TIMEWISE"] > stable_max:
            # high risk and Improving
            return 2
        else:
            # high risk stable
            return 3
    else:
        if row[var_name + "_SLOPE_TIMEWISE"] < stable_min:
            # Low risk unstable
            return 1
        elif row[var_name + "_SLOPE_TIMEWISE"] > stable_max:
            # low risk unstable
            return 1
        else:
            # Stable
            return 0


def create_features(icu_df: pd.DataFrame, periods: int, verbose: bool = True) -> pd.DataFrame:
    """ Preprocessing pipeline for ICU datasets"""
    print("Running feature engineering pipeline")
    return (
        icu_df.pipe(split_non_monotonic, settings.split_points)
              .pipe(create_time_delta)
              .pipe(create_ts_base_features, settings.standard_variables, periods=periods)
              .pipe(create_ts_slope_features,
                    settings.standard_variables,
                    periods=periods,
                    create_slope_categories=True,
                    stable_thresholds=settings.stable_24hr_slope_min_max
              )
    )


def get_all_feature_names():
    """ Return list of engineered features"""
    # split features
    pos_split_features = [x + "_POS" for x in settings.split_points.keys()]
    neg_split_features = [x + "_NEG" for x in settings.split_points.keys()]
    split_features = pos_split_features + neg_split_features
    # standard features (unsplit)
    standard_features = list(set(settings.standard_variables) - set(settings.split_points.keys()))
    # timeseries features
    ts_features = [
        f'{var}_{f}' for var in settings.standard_variables
        for f in ['DIFF', 'ROLAVG', 'ROLSTD', 'ALLAVG', 'ALLSTD']
    ]
    # slope features
    slope_features = [f"{var}_SLOPE" for var in settings.standard_variables]
    slope_timewise_features = [f"{var}_SLOPE_TIMEWISE" for var in settings.standard_variables]
    slope_cat_features = [f"{var}_SLOPE_CAT" for var in settings.stable_24hr_slope_min_max.keys()]
    return split_features + standard_features + ts_features + slope_cat_features