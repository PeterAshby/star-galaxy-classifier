import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from src.visualisation.plots import plot_confusion_matrix

def make_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
def train_model(model, abbrv):
    data_path = r'C:\Users\pmash\PycharmProjects\star-galaxy-classifier\data\sdss_processed_data.csv'
    df = pd.read_csv(data_path)
    df = df.drop(columns=['objid', 'ra', 'dec', 'type'])


    le = LabelEncoder()
    X = df.drop(columns='class')
    y = le.fit_transform(df['class'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99, stratify=y)
    pipeline = make_pipeline(model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, title=f'{abbrv} Conf-Matrix')

train_model(LogisticRegression(), 'LR')
