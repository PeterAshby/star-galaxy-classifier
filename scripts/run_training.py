import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from src.models.model_selection import evaluate_model
from src.models.train_model import train_model
import joblib


def run_training(models):
    data_path = Path(__file__).resolve().parents[1] / "data" / "sdss_processed_data.csv"
    df = pd.read_csv(data_path)
    df = df.drop(columns=['objid', 'ra', 'dec', 'type'])

    le = LabelEncoder()
    X = df.drop(columns='class')
    y = le.fit_transform(df['class'])

    trained_models = {}
    results = {}

    for abbrv, model in models.items():
        trained_model= train_model(model, X, y, abbrv)
        trained_models[abbrv] = trained_model
        results[abbrv] = evaluate_model(trained_model, X, y, 5, abbrv)
    for abbrv, metric in results.items():
        print(f"\n{abbrv} - Score: {results[abbrv]['mean_score']:.2f} ± {results[abbrv]['std_score']:.4f} | F1 Score: {results[abbrv]['f1_score']:.2f}")

if __name__ == '__main__':
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models = {
        'LR-best': joblib.load(models_dir / 'LR_best_model.pkl'),
        'RF-best': joblib.load(models_dir / 'RF_best_model.pkl'),
        'XGB-best': joblib.load(models_dir / 'XGB_best_model.pkl'),
        'SVM-best': joblib.load(models_dir / 'SVM_best_model.pkl')
    }
    run_training(models)