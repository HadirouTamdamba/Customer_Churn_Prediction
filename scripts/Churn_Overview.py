import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Upload data 
data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Data Overview
print(data.head())
print(data.info())
print(data.describe())

# Distribution of target variable
sns.countplot(x='Churn', data=data)
plt.title('Churn distribution')
plt.show() 

# Correlations
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()




