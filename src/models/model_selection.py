from sklearn.model_selection import cross_val_score, StratifiedKFold
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