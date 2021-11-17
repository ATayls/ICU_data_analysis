from utils import load_data
from preprocessing import preprocess
from feature_engineering import create_features, get_all_feature_names
from settings import PROC_DATA_DIR, SAVED_RESULTS_DIR, PLOTS_DIR
from news2_functions import bootstrap_news2
from train import train_logistic_model_bootstrapped, train_logistic_model_CV_grouped, train_logistic_model_cv, run_lr_train
from plots import compare_cv_results, shap_linear_summary, permutation_importance_plot

import pandas as pd
from sklearn.preprocessing import StandardScaler


################################################################################
## Extract and Transform function
################################################################################

def ETL(filename: str, data_version: str, ts_n_obs: int):
    """
    Handles loading, preprocessing and feature engineering of data.
    Loads existing base on data version.
    :param filename: filename of data in settings.DATA_DIR
    :param data_version: tracks preproc / FE changes
    :param ts_n_obs: number of observations to use in timeseries calculations
    :return:
    """
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


def main(
    filename_train, filename_test, dependant_var, ts_n_obs, data_version, n_bootstraps
):
    """
    Main experiment function
    :param filename_train: filename of training data in settings.DATA_DIR
    :param filename_test: filename of testing data in settings.DATA_DIR
    :param dependant_var: target variable
    :param ts_n_obs: number of observations to use in timeseries calculations
    :param data_version: data versioning number
    :param n_bootstraps: number of bootstraps
    :return:
    """
    print(f"RUN:\n--TRAIN:{filename_train}\n--TEST:{filename_test}")
    ##########################################################
    ## Load data
    ##########################################################

    df_tr = ETL(filename_train, data_version, ts_n_obs)
    df_te = ETL(filename_test, data_version, ts_n_obs)

    ##########################################################
    ## Modelling
    ##########################################################

    feature_list = get_all_feature_names()
    news2_results_tr = bootstrap_news2(df_tr, n_bootstraps, dependant_var, threshold=5)
    news2_results_te = bootstrap_news2(df_te, n_bootstraps, dependant_var, threshold=5)

    results_dict_bs = train_logistic_model_bootstrapped(
        df_tr, feature_list, dependant_var, n_bootstraps, fpr_match=1-0.936
    )
    results_dict_cv = train_logistic_model_CV_grouped(
        df_tr, feature_list, dependant_var, groups=df_tr["ADMISSION_ID"], folds=10, fpr_match=1-0.936
    )

    test_results_dict = run_lr_train(
        df_tr[feature_list], df_te[feature_list], df_tr[DEPENDANT_VAR], df_te[DEPENDANT_VAR]
    )
    print(f"DEWS TEST RESULTS: AUROC:{test_results_dict['metrics']['AUC ROC']} "
          f"AUPRC:{test_results_dict['metrics']['AUC PR']}")
    print(f"NEWS2 TEST RESULTS: AUROC:{news2_results_te['CV_AVG']['metrics']['AUC ROC']} "
          f"AUPRC:{news2_results_te['CV_AVG']['metrics']['AUC PR']}")

    ##########################################################
    ## Plots
    ##########################################################

    compare_cv_results(news2_results_tr, results_dict_cv)

    permutation_importance_plot(
        results_dict_cv, feature_list,
        title="Logistic Regression Feature Importance (10FoldCV)"
    )

    model_cv1 = test_results_dict['model']
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_tr[feature_list])
    shap_linear_summary(model_cv1, data_scaled, feature_list)


if __name__ == '__main__':

    ################################################################################
    ## Run configuration
    ################################################################################

    # Run config
    DATA_VERSION = "1"
    FILENAME_TRAIN = 'Respiratory admissions April 2015 to December 2019 excel v11_anonymised.xlsx'
    FILENAME_TEST = 'Respiratory admissions January 2020 to December 2020 v1_anonymised.xlsx'
    TS_N_OBS = 5
    DEPENDANT_VAR = "24_HOURS_FROM_EVENT"
    N_BOOTSTRAPS = 50

    main(FILENAME_TRAIN, FILENAME_TEST, DEPENDANT_VAR, TS_N_OBS, DATA_VERSION, N_BOOTSTRAPS)

    print("")