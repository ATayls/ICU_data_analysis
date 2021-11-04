import pandas as pd
import settings


def load_data(filename: str) -> pd.DataFrame:
    """Load raw data as dataframe. Stores as pickle to reduce future loading times"""
    extension = filename.split(".")[-1]
    raw_path = settings.RAW_DATA_DIR.joinpath(filename)
    pickle_path = settings.RAW_DATA_DIR_PKL.joinpath(filename).with_suffix(".pkl")
    if pickle_path.exists():
        print(f"Loading {filename} from pickle file")
        df = pd.read_pickle(pickle_path)
    elif raw_path.exists():
        if extension == "csv":
            print(f"Loading {filename}")
            df = pd.read_csv(raw_path)
        elif extension == "xlsx":
            print(f"Loading {filename}")
            df = pd.read_excel(raw_path)
        else:
            raise ValueError(f"Unrecognised Filetype {extension}")
        print("Saving as Pickle")
        df.to_pickle(pickle_path)
    else:
        raise ValueError(f"{raw_path} not found.")
    return df
