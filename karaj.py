#!/usr/bin/env python
# coding: utf-8

# In[5]:


# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Barghozari dade-ha
file_path = 'file:///C:/Users/amir-sol/Desktop/alborz3.xlsx' 
data = pd.read_excel(file_path)

# Hozf faza-ye ezafi az nam-e soton-ha
data.columns = data.columns.str.strip()

# Nam-e soton-ha
print(data.columns)

# Tedad maqadir ghomshode dar har soton:
print("Tedad maqadir ghomshode dar har soton:")
print(data.isnull().sum())

# Dictionary tarjome nam-e shahrha
city_translation = {
    'طالقان': 'Taleqan',
    'نظرآباد': 'Nazareabad',
    'ساوجبلاغ': 'Savojbolagh',
    'کرج': 'Karaj',
    'کل': 'Total Cities',
    'مهاجر': 'Mohajer'
}

# Shahrha-ye mored nazar
cities_of_interest = ['Taleqan', 'Nazareabad', 'Savojbolagh', 'Karaj', 'Total Cities']

# Mohasebe tedad va miaangin vorood afrad be shahrha
total_counts = data[cities_of_interest].sum()
mean_counts = data[cities_of_interest].mean()

print("Tedad afradi ke be har yeki az shahrha amade-and:")
print(total_counts)
print("Miaangin tedad afradi ke be har yeki az shahrha amade-and:")
print(mean_counts)

# Tedad afradi ke az har shahr be shahrha-ye mored nazar amade-and
incoming_counts = data.groupby('Mohajer')[cities_of_interest].sum()
print("Tedad afradi ke az har shahr be shahrha-ye mored nazar amade-and:")
print(incoming_counts)

# Shahr ba bishtarin tedad vorood
max_entry_city = total_counts.idxmax()
max_entry_value = total_counts.max()
print(f"Shahr ba bishtarin tedad vorood: {max_entry_city} ba {max_entry_value} nafar")

# Tosee amari dade-ha
print(data[cities_of_interest].describe())

# Jabe-nemoodar tozi-e tedad afrad be shahrha-ye mokhtalef
plt.figure(figsize=(10, 6))
data[cities_of_interest].plot(kind='box')
plt.title('Jabe-nemoodar tozi-e tedad afrad be shahrha-ye mokhtalef')
plt.ylabel('Tedad afrad')
plt.grid(axis='y')
plt.show()

# Namoodar mile-i tedad afradi ke be har yeki az shahrha amade-and
plt.figure(figsize=(10, 6))
total_counts.plot(kind='bar', color='skyblue')
plt.title('Tedad afradi ke be har yeki az shahrha amade-and')
plt.xlabel('Shahrha')
plt.ylabel('Tedad afrad')
plt.xticks(ticks=range(len(cities_of_interest)), labels=cities_of_interest, rotation=45)  
plt.grid(axis='y')
plt.show()

# Mohasebe hambastegi beyn shahrha
correlation_matrix = data[cities_of_interest].corr()
print("Hambastegi beyn shahrha:")
print(correlation_matrix)

# Naghshe hararti hambastegi beyn shahrha
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Naghshe hararti hambastegi beyn shahrha')
plt.show()

# Shenasayi maqadir port
outliers = {}
for city in cities_of_interest:
    mean = data[city].mean()
    std_dev = data[city].std()
    outlier_condition = data[city] > (mean + std_dev)
    outliers[city] = data[outlier_condition]

print("Maqadir port baraye har shahr:")
for city, outlier_data in outliers.items():
    print(f"{city}:\n{outlier_data}\n")

# Jabe-nemoodar baraye maqadir port
num_cities = len(outliers)
rows = (num_cities + 1) 
plt.figure(figsize=(10, 6))
for i, city in enumerate(outliers.keys()):
    plt.subplot(rows, 2, i + 1) 
    plt.boxplot(outliers[city][city].dropna())
    plt.title(f'Jabe-nemoodar baraye {city}')  
    plt.ylabel('Tedad afrad')
plt.tight_layout()
plt.show()

# Hozf maqadir port az dade-ha
cleaned_data = data[~data.index.isin(pd.concat(outliers.values()).index)]

# Mohasebe miaangin va enhedaf ma'yar ghabl va ba'ad az huzoof maqadir port
mean_before = data[cities_of_interest].mean()
std_before = data[cities_of_interest].std()
mean_after = cleaned_data[cities_of_interest].mean()
std_after = cleaned_data[cities_of_interest].std()

print("Miaangin va enhedaf ma'yar ghabl va ba'ad az huzoof maqadir port:")
print("Ghabl az huzoof maqadir port:")
print(mean_before, std_before)
print("Ba'ad az huzoof maqadir port:")
print(mean_after, std_after)

# Barresi vojud soton-ha-ye 'Mohajer' va 'Karaj'
if 'Mohajer' in data.columns and 'Karaj' in data.columns:
    data = data.dropna(subset=['Mohajer', 'Karaj'])

    # Kodguzari maqadir mohajer
    data['Mohajer'] = data['Mohajer'].astype(str)  
    data['Mohajer_encoded'] = data['Mohajer'].factorize()[0]  

    # Tayin viizgi-ha va barcheb-ha
    X = data[['Mohajer_encoded']]  
    y = data['Karaj']  

    # Taqsim dade-ha be majmooe-haye amoozeshi va azmayeshi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sakht va amoozesh model regreshan khatti
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Pishbini maqadir
    predictions = model.predict(X_test)

    # Rasme namoodar maqadir va pishbini shode
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, predictions, color='red', label='Predicted')
    plt.title('Actual vs Predicted for tedad mohajeran be Karaj')
    plt.xlabel('Mohajer (kodguzari shode)')
    plt.ylabel('Tedad mohajeran be Karaj')
    plt.legend()
    plt.show()
else:
    print("Soton-ha-ye 'Mohajer' va 'Karaj' dar dade-ha vojud nadarand.")


# In[ ]:




