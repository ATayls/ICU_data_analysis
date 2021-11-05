from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Raw data folders
RAW_DATA_DIR = ROOT_DIR.joinpath(r'raw_data/raw/')
RAW_DATA_DIR_PKL = ROOT_DIR.joinpath(r'raw_data/pickled/')

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
    "SYSTOLIC_BP": (-10, 10),
    "HEART_RATE": (-10, 10),
    "RESP_RATE": (-2, 2),
    "O2_SATS": (-2, 2),
    "INSPIRED_O2_LITRES": (-0.5, 0.5),
    "INSPIRED_O2_%": (-3, 3)
}

slope_categories = {
    "stable"
}