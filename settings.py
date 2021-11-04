from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Raw data folders
RAW_DATA_DIR = ROOT_DIR.joinpath(r'raw_data/raw/')
RAW_DATA_DIR_PKL = ROOT_DIR.joinpath(r'raw_data/pickled/')
