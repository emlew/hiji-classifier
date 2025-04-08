from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def get_vars():
    # return ['accelX_mean', 'accelX_var', 
    #         'accelX_min', 'accelY_mean', 
    #         'accelY_max', 'accelY_min', 
    #         'accelZ_mean', 'accelZ_max', 
    #         'accelZ_min', 'rateX_max', 
    #         'rateX_min', 'rateY_min', 
    #         'rateZ_min']
    # return ['accelX_mean', 'accelX_var', 'accelX_min', 'accelY_mean',
    #         'accelY_max', 'accelY_min', 'accelZ_mean', 'accelZ_max', 'accelZ_min', 
    #          'rateX_max', 'rateX_min', 
    #          'rateY_min','rateZ_min']
    # return ['accelX_mean', 'accelX_var', 
    #         'accelY_max', 'accelY_min', 'accelZ_mean', 'accelZ_max', 
    #         'rateX_max', 'rateX_min',
    #         'rateZ_var', ]
    # return ['accelX_min','accelY_mean', 
    #    'accelY_max', 'accelY_min', 'accelZ_mean', 'accelZ_max', 'accelZ_min', 
    #    'rateX_mean', 'rateX_var','rateX_max', 'rateX_min', 'rateY_mean', 'rateY_var', 
    #    'rateY_max', 'rateY_min', 'rateZ_var', 'rateZ_max']
    return ['accelX_max', 'accelX_min',
       'accelY_var', 'accelY_max', 'accelY_min', 
       'accelZ_var',  'rateX_var',
       'rateX_max', 'rateY_mean', 'rateY_var']

def get_split(filename, use_alt_preprocessing=False):
    df = pd.read_csv(filename)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if use_alt_preprocessing:
        X = df[['resLA_median','resLA_iqr','resAA_median','resAA_iqr']]
    else:
        # Initialize StandardScaler
        scaler = StandardScaler()

        df[['accelX_mean', 'accelX_var', 'accelX_max', 'accelX_min',
        'accelY_mean', 'accelY_var', 'accelY_max', 'accelY_min', 'accelZ_mean',
        'accelZ_var', 'accelZ_max', 'accelZ_min', 'rateX_mean', 'rateX_var',
        'rateX_max', 'rateX_min', 'rateY_mean', 'rateY_var', 'rateY_max',
        'rateY_min', 'rateZ_mean', 'rateZ_var', 'rateZ_max', 'rateZ_min']] = scaler.fit_transform(df[['accelX_mean', 'accelX_var', 'accelX_max', 'accelX_min',
        'accelY_mean', 'accelY_var', 'accelY_max', 'accelY_min', 'accelZ_mean',
        'accelZ_var', 'accelZ_max', 'accelZ_min', 'rateX_mean', 'rateX_var',
        'rateX_max', 'rateX_min', 'rateY_mean', 'rateY_var', 'rateY_max',
        'rateY_min', 'rateZ_mean', 'rateZ_var', 'rateZ_max', 'rateZ_min']])
        
        X = df[get_vars()]
    y = df['impact']

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
    return X_train, X_test, y_train, y_test

def print_evaluation(y_test, y_pred, y_prob):
    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Print evaluation results
    print("Test Accuracy:", accuracy)
    print("Test Precision:", precision)
    print("Test Recall:", recall)
    print("Test F1 Score:", f1)
    print("Test ROC-AUC Score:", roc_auc)

    # Print confusion matrix and classification report
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
def plot_confusion_matrix(y_test, y_pred):
    # Plot Confusion Matrix in a prettier way
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.show()