from utils import load_data
from preprocessing import preprocess
from feature_engineering import create_features, get_all_feature_names
from settings import PROC_DATA_DIR
from news2_functions import bootstrap_news2
from train import train_logistic_model_cv

import pandas as pd

################################################################################
## Run configuration
################################################################################

# Run config
DATA_VERSION = "1"
FILENAME = 'Annotated_dataset_training_anonymised_V2.xlsx'
TS_N_OBS = 5
DEPENDANT_VAR = "24_HOURS_FROM_EVENT"
N_BOOTSTRAPS = 150

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

feature_list = get_all_feature_names()

##########################################################
## Modelling
##########################################################

news2_bootstrapped_auc = bootstrap_news2(df, N_BOOTSTRAPS, DEPENDANT_VAR)
dews2_cv_results = train_logistic_model_cv(df, feature_list, DEPENDANT_VAR, folds=10, fpr_match=1-0.936)

print("")
