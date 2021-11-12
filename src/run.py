from utils import load_data
from preprocessing import preprocess
from feature_engineering import create_features, get_all_feature_names
from settings import PROC_DATA_DIR
from news2_functions import bootstrap_news2
from train import train_logistic_model_bootstrapped, train_logistic_model_CV_grouped

import pandas as pd

################################################################################
## Run configuration
################################################################################

# Run config
DATA_VERSION = "1"
FILENAME_TRAIN = 'Annotated dataset_validation_anonymised.xlsx'
FILENAME_TEST = 'Annotated dataset_validation_anonymised.xlsx'
TS_N_OBS = 5
DEPENDANT_VAR = "4_HOURS_FROM_ANNOTATED_EVENT"
N_BOOTSTRAPS = 300

################################################################################
## Extract and Transform function
################################################################################

def ETL(filename: str, data_version: str, ts_n_obs: int):
    processed_fp = PROC_DATA_DIR.joinpath(
        f"V{data_version}/{ts_n_obs}OBS_{filename.split('.')[0]}.pkl"
    )
    # Load or create processed dataset.
    if not processed_fp.exists():
        processed_fp.parent.mkdir(parents=True, exist_ok=True)
        df = load_data(filename)
        df = preprocess(df)
        df = create_features(df, periods=ts_n_obs)
        df.to_pickle(processed_fp)
    else:
        df = pd.read_pickle(processed_fp)
    return df

##########################################################
## Load data
##########################################################

df_tr = ETL(FILENAME_TRAIN, DATA_VERSION, TS_N_OBS)
df_te = ETL(FILENAME_TEST, DATA_VERSION, TS_N_OBS)

##########################################################
## Modelling
##########################################################
feature_list = get_all_feature_names()
news2_bootstrapped_auc = bootstrap_news2(df_te, N_BOOTSTRAPS, DEPENDANT_VAR, threshold=5)

dews2_bs_results = train_logistic_model_bootstrapped(
    df_tr, feature_list, DEPENDANT_VAR, N_BOOTSTRAPS, fpr_match=1-0.936,
    test_icu_df=df_te
)
dews2_cvg_results = train_logistic_model_CV_grouped(
    df_tr, feature_list, DEPENDANT_VAR, groups=df_tr["ADMISSION_ID"], folds=10, fpr_match=1-0.936
)

print("")
