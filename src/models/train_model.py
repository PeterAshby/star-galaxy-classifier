import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.visualisation.plots import plot_confusion_matrix
from pathlib import Path


def make_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

def train_model(model, X, y, abbrv):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99, stratify=y)
    pipeline = make_pipeline(model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, title=f'{abbrv} Conf-Matrix')

    return pipeline