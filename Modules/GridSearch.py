from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
import numpy as np

def grid_search(base_model, X_train, y_train):
    param_grid = {
        'C': [1, 10, 100, 1000],
        'gamma': ['auto', 1, 0.1, 0.001, 0.001],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, return_train_score=True)
    grid_search.fit(X_train, y_train)

    return grid_search