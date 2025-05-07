from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, f1_score
import numpy as np

def evaluate_model(model, X, y, cv=4, abbrv=''):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=99)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1 = cross_val_score(model, X, y, cv=skf, scoring=make_scorer(f1_score, average='macro'))

    results = {
        'mean_score': np.mean(scores),
        'std_score' : np.std(scores),
        'f1_score' : np.mean(f1),
        'raw_scores': scores
    }
    return results

def grid_search_model(pipeline, params, X, y, cv=3, scoring='accuracy'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99, stratify=y)
    grid = GridSearchCV(pipeline, params, cv=cv, scoring=scoring, n_jobs=-1)
    grid.fit(X_train, y_train)

    return grid.best_params_, grid.best_estimator_, grid.best_score_