from utils import load_data
from preprocessing import preprocess
from feature_engineering import create_features
from settings import PROC_DATA_DIR

import pandas as pd

################################################################################
## Run configuration
################################################################################

# Run config
DATA_VERSION = "1"
FILENAME = 'Annotated_dataset_training_anonymised_V2.xlsx'
TS_N_OBS = 5

################################################################################
## Extract and Transform
################################################################################

processed_fp = PROC_DATA_DIR.joinpath(
    f"V{DATA_VERSION}/{TS_N_OBS}OBS_{FILENAME.split('.')[0]}.pkl"
)

# Load or create processed dataset.
if not processed_fp.exists():
    processed_fp.parent.mkdir(parents=True, exist_ok=True)
    df = load_data(FILENAME)
    df = preprocess(df)
    df = create_features(df, periods=TS_N_OBS)
    df.to_pickle(processed_fp)
else:
    df = pd.read_pickle(processed_fp)

##########################################################
## Modelling
##########################################################

print("")
