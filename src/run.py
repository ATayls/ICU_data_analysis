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
    print(f"\nA) Calculating NEWS2 results:")
    news2_results_tr = bootstrap_news2(df_tr, n_bootstraps, dependant_var, threshold=5)
    news2_results_te = bootstrap_news2(df_te, n_bootstraps, dependant_var, threshold=5)

    print(f"\nB) Training on {filename_train}, validating with bootstrap & 10foldCV:")
    results_dict_bs_tr = train_logistic_model_bootstrapped(
        df_tr, feature_list, dependant_var, n_bootstraps, fpr_match=1-0.936
    )
    results_dict_cv_tr = train_logistic_model_CV_grouped(
        df_tr, feature_list, dependant_var, groups=df_tr["ADMISSION_ID"], folds=10, fpr_match=1-0.936
    )

    print(f"\nC) Training on {filename_train}, testing on {filename_test} bootstrapping for confidence intervals")
    results_dict_bs_te = train_logistic_model_bootstrapped(
        df_tr, feature_list, dependant_var, n_bootstraps, fpr_match=1-0.936,
        test_icu_df=df_te
    )

    print(f"\nTraining on FULL DATASET {filename_train} , testing on {filename_test}")
    test_results_dict = run_lr_train(
        df_tr[feature_list], df_te[feature_list], df_tr[DEPENDANT_VAR], df_te[DEPENDANT_VAR]
    )
    print(f"FULL MODEL DEWS TEST RESULTS: AUROC:{test_results_dict['metrics']['AUC ROC']} "
          f"AUPRC:{test_results_dict['metrics']['AUC PR']}")
    print(f"FULL MODEL NEWS2 TEST RESULTS: AUROC:{news2_results_te['CV_AVG']['metrics']['AUC ROC']} "
          f"AUPRC:{news2_results_te['CV_AVG']['metrics']['AUC PR']}")

    ##########################################################
    ## Plots
    ##########################################################

    compare_cv_results(news2_results_tr, results_dict_bs_tr)
    compare_cv_results(news2_results_te, results_dict_bs_te)

    permutation_importance_plot(
        results_dict_cv_tr, feature_list,
        title="Logistic Regression Feature Importance (10FoldCV)",
        save_path=SAVED_RESULTS_DIR.joinpath(f"FI_{filename_train}.csv")
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
    DATA_VERSION = "2"
    N_BOOTSTRAPS = 250
    TS_N_OBS = 5

    FILENAME_TRAIN = 'Respiratory admissions April 2015 to December 2019 excel v11_anonymised.xlsx'
    FILENAME_TEST = 'Respiratory admissions January 2020 to December 2020 v1_anonymised.xlsx'
    DEPENDANT_VAR = "24_HOURS_FROM_EVENT"
    main(FILENAME_TRAIN, FILENAME_TEST, DEPENDANT_VAR, TS_N_OBS, DATA_VERSION, N_BOOTSTRAPS)

    FILENAME_TRAIN = 'Annotated_dataset_training_anonymised_V2.xlsx'
    FILENAME_TEST = 'Annotated dataset_validation_anonymised.xlsx'
    DEPENDANT_VAR = "4_HOURS_FROM_ANNOTATED_EVENT"
    main(FILENAME_TRAIN, FILENAME_TEST, DEPENDANT_VAR, TS_N_OBS, DATA_VERSION, N_BOOTSTRAPS)

    print("")