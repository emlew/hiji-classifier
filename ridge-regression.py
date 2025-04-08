from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

df = pd.read_csv('./final_preprocessed_data.csv')
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Separate features (X) and target (y)
X = df[['accelX_mean', 'accelX_var', 'accelX_max', 'accelX_min',
       'accelY_mean', 'accelY_var', 'accelY_max', 'accelY_min', 'accelZ_mean',
       'accelZ_var', 'accelZ_max', 'accelZ_min', 'rateX_mean', 'rateX_var',
       'rateX_max', 'rateX_min', 'rateY_mean', 'rateY_var', 'rateY_max',
       'rateY_min', 'rateZ_mean', 'rateZ_var', 'rateZ_max', 'rateZ_min']]
y = df['impact']

# Standardize independent variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Ridge Regression model with alpha (λ = regularization strength)
ridge = Ridge(alpha=1000.0)

# Fit the model
ridge.fit(X_train, y_train)

# Print model coefficients
print("Ridge Coefficients:", ridge.coef_)

# Predict on test set
y_pred = ridge.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# # Cross-validation with different alpha values
# ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 10), store_cv_values=True)
# ridge_cv.fit(X_train, y_train)

# # Best alpha
# print("Best alpha:", ridge_cv.alpha_)

from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(alphas=np.logspace(-3, 3, 10), cv=5)
lasso_cv.fit(X_train, y_train)

print("Best alpha for Lasso:", lasso_cv.alpha_)
print("Lasso Coefficients:", lasso_cv.coef_)
