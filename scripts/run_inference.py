import pandas as pd
import pickle
from pathlib import Path
from src.data.preprocessing import process_sdss_data
from src.data.sdss_query import get_sdss_data


def load_model(model_name):
    model_path = Path(__file__).resolve().parents[1] / 'models' / f'{model_name}_best_model.pkl'
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def run_inference(model, data):
    X = data.drop(columns='class')
    preds = model.predict(X)
    return preds

if __name__ == '__main__':
    preds = run_inference('XGB', 2000)

    print(preds)