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


def drop_first_n_observations(df: pd.DataFrame, identifier: str, n_obs: int) -> pd.DataFrame:
    """ Drops the first n obs for each identifier group. Assumes input dataframe is ordered."""

    df["__COUNT__"] = 1
    df["__CUMSUM__"] = (
        df.groupby(identifier)["__COUNT__"]
            .cumsum()
    )
    return df[~df["__CUMSUM__"].isin(list(range(1, n_obs+1)))].drop(
        columns=["__CUMSUM__", "__COUNT__"]
    ).reset_index(drop=True)