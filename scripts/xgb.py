from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from tools.train_test import get_split, print_evaluation

"""
This script performs a grid search to find the best parameters for an XGB classifier. It also prints an evaluation of the model.
"""

X_train, X_test, y_train, y_test = get_split('./final_preprocessed_data.csv',True)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(XGBClassifier(eval_metric="logloss"), param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_  # Use best model from tuning
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

print_evaluation(y_test, y_pred, y_prob)
