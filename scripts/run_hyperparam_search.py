from pathlib import Path
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
from src.models.model_selection import grid_search_model
from src.models.train_model import make_pipeline


def hyperparameter_search(models, params):
    data_path = Path(__file__).resolve().parents[1] / "data" / "sdss_processed_data.csv"
    models_dir = Path(__file__).resolve().parents[1] / 'models'
    results_path = Path(__file__).resolve().parents[1] / "data" / "best_model_results.txt"
    df = pd.read_csv(data_path)
    df = df.drop(columns=['objid', 'ra', 'dec', 'type'])
    le = LabelEncoder()
    X = df.drop(columns='class')
    y = le.fit_transform(df['class'])

    best_models = {}
    with open(results_path, 'w') as f:
        for abbrv, model in models.items():
            param_grid = params[abbrv]
            pipeline = make_pipeline(model)
            best_params, best_model, best_score = grid_search_model(pipeline, param_grid, X, y)
            best_models[abbrv] = {
                'model': best_model,
                'params': best_params,
                'score': best_score
            }

            f.write(f'{abbrv}:\n')
            f.write(f"Best Score: {best_score:.4f}\n")
            f.write(f"Best Params: {best_params}\n\n")

            model_path = models_dir / f'{abbrv}_best_model.pkl'
            with open(model_path, 'wb') as model_file:
                pickle.dump(best_model, model_file)
    return best_models
models  = {
    'LR': LogisticRegression(),
    'RF': RandomForestClassifier(),
    'XGB': XGBClassifier(),
    'SVM': SVC()
}
params = {
    'LR': {
        'model__C': [0.01, 0.1, 1, 10, 100],      # Regularization strength
        'model__penalty': ['l2'],                 # 'l1' if solver='liblinear'
        'model__solver': ['lbfgs'],               # 'liblinear' if you want 'l1'
        'model__max_iter': [100, 500]
    },
    'RF': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2],
        'model__bootstrap': [True, False]
    },
    'XGB': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 6, 10],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__subsample': [0.8, 1],
        'model__colsample_bytree': [0.8, 1],
        'model__gamma': [0, 1]
    },
    'SVM': {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto']  # Only used for 'rbf'
    }
}

hyperparameter_search(models, params)
