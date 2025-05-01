import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

"""
This script conducts an ANOVA analysis of the data and tests for multicollinearity.
"""

df = pd.read_csv('./final_preprocessed_data.csv')
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Initialize StandardScaler
scaler = StandardScaler()

# Standardize independent variables (exclude the dependent variable 'y')
timestamp_cols = [col for col in df.columns if 'timestamp' in col]
X = df.drop(columns=timestamp_cols)
X.drop(columns=['isoTimestamp','impact'],inplace=True)
df[X.columns] = scaler.fit_transform(df[X.columns])

# Create string of regressors to be used in linear regression model
regressor_string = ' + '.join(X.columns)

# Define independent variables (add a constant for the intercept)
X = sm.add_constant(X)  # Adds the intercept term

# Define dependent variable
y = df['impact']

# Fit linear regression model
model = smf.ols('y ~ '+regressor_string, data=df).fit()

# Print summary
print(model.summary())

# Perform ANOVA
anova_results = sm.stats.anova_lm(model)
print(anova_results)

# Calculate VIF for each variable
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Print VIF results
print(vif_data)
print('Regressors at risk for multicollinearity: ')
print(vif_data[vif_data['VIF']>10])

# Compute correlation matrix
df = df.drop(columns=['isoTimestamp','timestamp_mean', 'timestamp_var', 'timestamp_max', 'timestamp_min','impact'])
corr_matrix = df.corr()

# Print correlation values
print(corr_matrix)

# Plot heatmap for better visualization
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()