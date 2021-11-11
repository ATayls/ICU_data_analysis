"""
File to contain all preprocessing functions for ICU data.
Entry Point: preprocess()
"""

import pandas as pd
import numpy as np


def parse_inspired_o2(icu_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """ Parse inspired o2 features """
    icu_df = icu_df.copy()
    # Split Inspired O2 into Inspired_O2_litres, Inspired_02_% for further analysis
    icu_df["INSPIRED_O2_%"] = icu_df["INSPIRED_O2"].copy()
    icu_df["INSPIRED_O2_LITRES"] = icu_df["INSPIRED_O2"].copy()
    icu_df.loc[icu_df["INSPIRED_O2_UNITS"] == '%', ["INSPIRED_O2_LITRES"]] = np.nan
    icu_df.loc[icu_df["INSPIRED_O2_UNITS"] == 'litres', ["INSPIRED_O2_%"]] = np.nan
    # Fill nan values with 0 oxygen
    icu_df["INSPIRED_O2_%"] = icu_df["INSPIRED_O2_%"].fillna(21.0)
    icu_df["INSPIRED_O2_LITRES"] = icu_df["INSPIRED_O2_LITRES"].fillna(0.0)
    if verbose:
        print("--Parsed O2_% and O2_LITRES")
    return icu_df


def encode_alert(icu_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """ Encode the ACVPU categorical variable to an ordinal int type"""
    icu_df["ACVPU_ENCODED"] = icu_df["ACVPU"].copy()
    avcpu_mapping = {
        "ACVPU_ENCODED": {
            "Alert": 0,
            "Confused": 1,
            "Responds to voice": 1,
            "Voice": 1,
            "Pain": 2,
            "Responds to pain": 2,
            "Unresponsive": 3
        }
    }
    icu_df = icu_df.replace(avcpu_mapping)
    if verbose:
        print("--Encoded ACVPU")
    return icu_df


def encode_age_range(icu_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """ Create the AGE variable based on AGE Range. Assuming center age of range."""
    icu_df["AGE"] = icu_df["AGE_RANGE"].str[0:2].astype(int) + 2
    if verbose:
        print("--Encoded AGE_RANGE")
    return icu_df


def handle_missing(icu_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """ Null value handling. Drop all null rows, and any uncomplete observations"""
    null_rows = icu_df.isnull().all(1)
    icu_df = icu_df[~null_rows]
    incomplete_rows = icu_df["COMPLETE_DATA"] != 1
    icu_df = icu_df[~incomplete_rows]
    if verbose:
        print(f"--Dropped {null_rows.sum()} NA rows")
        print(f"--Dropped {incomplete_rows.sum()} rows labeled as incomplete")
    return icu_df


def handle_temperature(icu_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Handle cases where na placeholders are within TEMPERATURE variable.
    Currently drops these observations and ensures TEMPERATURE is floating point.
    """
    temperature_na_placeholders = [
        "Refused", "Unable", "Refused Not Concerned", "Refused Concerned"
    ]
    mask = icu_df["TEMPERATURE"].isin(temperature_na_placeholders)
    if mask.any():
        icu_df = icu_df[~mask]
        if verbose:
            print(f"--Dropped {mask.sum()} TEMPERATURE NA placeholder rows")
    icu_df["TEMPERATURE"] = icu_df["TEMPERATURE"].astype(float)
    return icu_df


def preprocess(icu_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """ Preprocessing pipeline for ICU datasets"""
    print("Running preprocessing pipeline")
    return (
        icu_df.pipe(handle_missing, verbose)
              .pipe(parse_inspired_o2, verbose)
              .pipe(encode_alert, verbose)
              .pipe(handle_temperature, verbose)
              #.pipe(encode_age_range, verbose)
    )
