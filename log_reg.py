import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from tools.train_test import get_split, print_evaluation

X_train, X_test, y_train, y_test = get_split('./final_preprocessed_data.csv',use_alt_preprocessing=False)

# Define hyperparameter grid
param_grid = {'C': np.logspace(-4, 4, 10)}  # More values from very small to large

# Stratified k-fold ensures balanced classes in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression with Grid Search
log_reg = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)

grid_search = GridSearchCV(log_reg, param_grid, cv=cv, scoring='f1', n_jobs=-1)

# Train using cross-validation
grid_search.fit(X_train, y_train)

# Best model and hyperparameters
best_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Make predictions on the test set
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

print_evaluation(y_test, y_pred, y_prob)