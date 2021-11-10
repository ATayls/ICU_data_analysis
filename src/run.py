from utils import load_data
from preprocessing import preprocess
from feature_engineering import create_features

if __name__ == '__main__':
    df = load_data(r'Annotated_dataset_training_anonymised_V2.xlsx')
    df = preprocess(df)
    df = create_features(df, periods=5)

    print("")
