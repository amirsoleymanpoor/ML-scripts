#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Total number of samples
num_samples = 2000

# Create a DataFrame with random data
np.random.seed(0)  # For reproducible random data
data = {
    'Gender': np.random.choice(['Male', 'Female'], num_samples),
    'Race': np.random.choice(['White', 'Asian', 'Black', 'Other'], num_samples),
    'Sector': np.random.choice(['Academia', 'Business', 'Government', 'Nonprofit', 'Consultant'], num_samples),
    'Base_Salary': np.random.randint(50000, 200000, num_samples),  # Random base salary
    'Total_Income': np.random.randint(60000, 220000, num_samples),  # Random total income
    'Job_Satisfaction': np.random.choice(['Very Satisfied', 'Somewhat Satisfied', 'Not Satisfied'], num_samples),
    'Years_Experience': np.random.randint(1, 20, num_samples),  # Random years of experience
}

df = pd.DataFrame(data)

# Analyze job popularity
sector_counts = df['Sector'].value_counts()

# Display job popularity
print("Job Popularity:")
print(sector_counts)

# Plot a bar chart to show job popularity
plt.figure(figsize=(10, 6))
sns.barplot(x=sector_counts.index, y=sector_counts.values, palette='viridis')
plt.title('Job Popularity')
plt.xlabel('Job')
plt.ylabel('Number of Observations')
plt.xticks(rotation=45)
plt.show()

# Analyze salary by gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Base_Salary', data=df)
plt.title('Salary Comparison by Gender')
plt.xlabel('Gender')
plt.ylabel('Base Salary')
plt.show()

# Analyze salary by race
plt.figure(figsize=(10, 6))
sns.boxplot(x='Race', y='Base_Salary', data=df)
plt.title('Salary Comparison by Race')
plt.xlabel('Race')
plt.ylabel('Base Salary')
plt.show()

# Analyze the relationship between experience and salary
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Years_Experience', y='Base_Salary', data=df)
plt.title('Relationship Between Experience and Base Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Base Salary')
plt.show()

# Analyze job satisfaction by gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Job_Satisfaction', hue='Gender', data=df)
plt.title('Job Satisfaction Analysis by Gender')
plt.xlabel('Job Satisfaction')
plt.ylabel('Number of Observations')
plt.show()

# Salary distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Base_Salary'], bins=30, kde=True)
plt.title('Base Salary Distribution')
plt.xlabel('Base Salary')
plt.ylabel('Number of Observations')
plt.show()

# Predictive modeling with OLS
X = df[['Years_Experience']]
X = sm.add_constant(X)  # Adding a constant
y = df['Base_Salary']
model = sm.OLS(y, X).fit()
print(model.summary())

# Clustering analysis
from sklearn.cluster import KMeans

# Select features for clustering
features = df[['Years_Experience', 'Base_Salary']]
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(features)

# Display clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Years_Experience', y='Base_Salary', hue='Cluster', data=df, palette='viridis')
plt.title('Clustering Analysis Based on Experience and Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Base Salary')
plt.show()


# In[ ]:




