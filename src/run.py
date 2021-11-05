from utils import load_data
from preprocessing import preprocess
from feature_engineering import create_ts_base_features, create_time_delta, create_ts_slope_features, split_non_symetric
import settings

if __name__ == '__main__':
    df = load_data(r'Annotated_dataset_training_anonymised_V2.xlsx')
    df = preprocess(df)
    df = split_non_symetric(df, settings.split_points)
    df = create_time_delta(df)
    df = create_ts_base_features(df, settings.standard_variables, periods=5)
    df = create_ts_slope_features(df, settings.standard_variables, periods=5)
    print("")
