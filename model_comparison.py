import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from tools.train_test import get_split, get_vars

use_alt_preprocessing = True

df = pd.read_csv('./final_preprocessed_data.csv')
df = df.replace([np.inf, -np.inf], np.nan).dropna()
if use_alt_preprocessing:
    X = df[['resLA_median','resLA_iqr','resAA_median','resAA_iqr']]
else:
    X = df[get_vars()]
y = df['impact']
# imps = len(df[df['impact'] == 1])
# tot = len(df['impact'])
# print('total: '+str(tot))
# print('impacts: '+str(imps))
# print('ratio: '+str(imps/tot))

# Train a simple Random Forest to check feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Plot feature importance
importances = rf.feature_importances_
feature_names = X.columns
sns.barplot(x=importances, y=feature_names)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance for Head Impact Detection")
plt.show()

X_train, X_test, y_train, y_test = get_split('./final_preprocessed_data.csv',use_alt_preprocessing)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss"),
    "SVM": SVC()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"{name}: Accuracy = {acc:.4f}, F1 Score = {f1:.4f}")