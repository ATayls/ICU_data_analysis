from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Raw data folders
RAW_DATA_DIR = ROOT_DIR.joinpath(r'raw_data/raw/')
RAW_DATA_DIR_PKL = ROOT_DIR.joinpath(r'raw_data/pickled/')

# Processed data folders
PROC_DATA_DIR = ROOT_DIR.joinpath(r'processed_data')

standard_variables = [
    "TEMPERATURE",
    "SYSTOLIC_BP",
    "HEART_RATE",
    "RESP_RATE",
    "O2_SATS",
    "ACVPU_ENCODED",
    "INSPIRED_O2_LITRES",
    "INSPIRED_O2_%",
]

split_points = {
    "TEMPERATURE": (36.1, 38),
    "SYSTOLIC_BP": (111, 219),
    "HEART_RATE": (51, 90),
    "RESP_RATE": (12, 20),
}

stable_24hr_slope_min_max = {
    "TEMPERATURE": (-0.5, 0.5),
    "SYSTOLIC_BP": (-10.0, 10.0),
    "HEART_RATE": (-10.0, 10.0),
    "RESP_RATE": (-2.0, 2.0),
    "O2_SATS": (-2.0, 2.0),
    "INSPIRED_O2_LITRES": (-0.5, 0.5),
    "INSPIRED_O2_%": (-3.0, 3.0)
}

slope_categories = {
    "stable"
}

column_name_mapping = {
    "NEWS_2_SCORE": "NEWS_2"
}