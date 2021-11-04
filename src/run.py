from utils import load_data
from preprocessing import preprocess

if __name__ == '__main__':
    df = load_data(r'Annotated_dataset_training_anonymised_V2.xlsx')
    df = preprocess(df)
