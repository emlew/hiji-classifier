import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from tools.train_test import get_split

"""
This script compares performance of logistic regression, random forest, XGB, and SVM classifiers.
"""

use_alt_preprocessing = False

df = pd.read_csv('./final_preprocessed_data.csv')
df = df.replace([np.inf, -np.inf], np.nan).dropna()

timestamp_cols = [col for col in df.columns if 'timestamp' in col]
X = df.drop(columns=timestamp_cols)
X.drop(columns=['isoTimestamp','impact'],inplace=True)
y = df['impact']
imps = len(df[df['impact'] == 1])
tot = len(df['impact'])
print('total number of bins: '+str(tot))
print('number of bins with impacts: '+str(imps))

# Consider ratio of impacts to total bins. If it's too low, model params and test/train split need to be balanced
print('ratio: '+str(imps/tot))

X_train, X_test, y_train, y_test = get_split('./final_preprocessed_data.csv',use_alt_preprocessing)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": Ran