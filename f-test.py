import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./final_preprocessed_data.csv')
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Initialize StandardScaler
scaler = StandardScaler()

# Standardize independent variables (exclude the dependent variable 'y')
df[['accelX_mean', 'accelX_var', 'accelX_max', 'accelX_min',
       'accelY_mean', 'accelY_var', 'accelY_max', 'accelY_min', 'accelZ_mean',
       'accelZ_var', 'accelZ_max', 'accelZ_min', 'rateX_mean', 'rateX_var',
       'rateX_max', 'rateX_min', 'rateY_mean', 'rateY_var', 'rateY_max',
       'rateY_min', 'rateZ_mean', 'rateZ_var', 'rateZ_max', 'rateZ_min']] = scaler.fit_transform(df[['accelX_mean', 'accelX_var', 'accelX_max', 'accelX_min',
       'accelY_mean', 'accelY_var', 'accelY_max', 'accelY_min', 'accelZ_mean',
       'accelZ_var', 'accelZ_max', 'accelZ_min', 'rateX_mean', 'rateX_var',
       'rateX_max', 'rateX_min', 'rateY_mean', 'rateY_var', 'rateY_max',
       'rateY_min', 'rateZ_mean', 'rateZ_var', 'rateZ_max', 'rateZ_min']])

# Define independent variables (add a constant for the intercept)
# X = df[['accelX_mean', 'accelX_var', 'accelX_max', 'accelX_min',
#        'accelY_mean', 'accelY_var', 'accelY_max', 'accelY_min', 'accelZ_mean',
#        'accelZ_var', 'accelZ_max', 'accelZ_min', 'rateX_mean', 'rateX_var',
#        'rateX_max', 'rateX_min', 'rateY_mean', 'rateY_var', 'rateY_max',
#        'rateY_min', 'rateZ_mean', 'rateZ_var', 'rateZ_max', 'rateZ_min']]
X = df[['accelX_var', 'accelX_max', 'accelY_max', 'accelY_min', 'accelZ_max', 'rateX_var',
       'rateX_max', 'rateY_var', 'rateY_min', 'rateZ_max', 'rateZ_min']]
X = sm.add_constant(X)  # Adds the intercept term

# Define dependent variable
y = df['impact']

# Fit linear regression model
# model = smf.ols('y ~ accelX_mean + accelX_var + accelX_max + accelX_min +  accelY_mean + accelY_var + accelY_max + accelY_min + accelZ_mean + accelZ_var + accelZ_max + accelZ_min + rateX_mean + rateX_var + rateX_max + rateX_min + rateY_mean + rateY_var + rateY_max + rateY_min + rateZ_mean + rateZ_var + rateZ_max + rateZ_min', data=df).fit()
# model = smf.ols('y ~ accelX_var + accelX_max + accelY_max + accelY_min + accelZ_max + rateX_var + rateX_max + rateY_var + rateY_min + rateZ_max + rateZ_min', data=df).fit()
model = smf.ols('y ~ accelX_mean + accelX_min + accelY_max + accelZ_max + accelZ_min + rateX_var + rateX_max + rateY_mean + rateY_min', data=df).fit()

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