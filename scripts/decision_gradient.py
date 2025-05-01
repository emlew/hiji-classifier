import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import GradientBoostingClassifier

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from tools.train_test import get_split

"""
This script performs a grid search to find the best parameters for decision tree and gradient boosting and prints an evaluation of the models.
"""

X_train, X_test, y_train, y_test = get_split('./final_preprocessed_data.csv', use_alt_preprocessing=False)

# Define hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 10, None],  # Limit tree depth to prevent overfitting
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 5]  # Minimum samples per leaf
}

# Perform Grid Search
dt = DecisionTreeClassifier(class_weight='balanced', random_state=22)
grid_search_dt = GridSearchCV(dt, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)

# Best model
best_dt = grid_search_dt.best_estimator_
print("Best Decision Tree Parameters:", grid_search_dt.best_params_)

y_pred_dt = best_dt.predict(X_test)
print("Decision Tree Performance:\n", classification_report(y_test, y_pred_dt))

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 300],  # Number of trees
    'learning_rate': [0.01, 0.1, 0.3],  # Step size for updates
    'max_depth': [3, 5],  # Depth of each tree
}

# Perform Grid Search
gb = GradientBoostingClassifier(random_state=42)
grid_search_gb = GridSearchCV(gb, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search_gb.fit(X_train, y_train)

# Best model
best_gb = grid_search_gb.best_estimator_
print("Best Gradient Boosting Parameters:", grid_search_gb.best_params_)

y_pred_gb = best_gb.predict(X_test)
print("Gradient Boosting Performance:\n", classification_report(y_test, y_pred_gb))

# Compute scores
dt_f1 = f1_score(y_test, y_pred_dt)
gb_f1 = f1_score(y_test, y_pred_gb)

print(f"Decision Tree F1 Score: {dt_f1:.3f}")
print(f"Gradient Boosting F1 Score: {gb_f1:.3f}")
