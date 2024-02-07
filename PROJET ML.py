""""#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Lire le fichier CSV
file_name = 'DataCoSupplyChainDataset.csv'
df = pd.read_csv(file_name, encoding='latin-1')

# Afficher les premières lignes du dataframe
print(df.head())

# Afficher les types de données
print(df.dtypes)

# Afficher les valeurs uniques dans la colonne 'Type'
print(df['Type'].value_counts())

# Afficher la forme du dataframe
print(df.shape)

# Afficher le nombre de valeurs manquantes par colonne
print(df.isnull().sum())

# Supprimer certaines colonnes
df.drop(columns=['Product Description', 'Order Zipcode'], axis=1, inplace=True)

# Afficher des informations sur le dataframe
print(df.info())

# Afficher les noms des colonnes
print(df.columns)

# Afficher les valeurs uniques dans la colonne 'Late_delivery_risk'
print(df['Late_delivery_risk'].value_counts())

# Afficher les valeurs uniques dans la colonne 'Order Item Discount'
print(df['Order Item Discount'].value_counts())

# Afficher un graphique de régression
sns.regplot(x='Sales', y='Order Item Discount', data=df)
plt.show()

# Supprimer d'autres colonnes
df.drop(columns=['Customer Email', 'Customer Fname', 'Customer Id', 'Customer Lname', 'Customer Password',
                 'Customer Street', 'Order City', 'Order Customer Id', 'order date (DateOrders)', 'Order Id',
                 'Order Item Cardprod Id'], axis=1, inplace=True)

# Afficher les noms des colonnes après suppression
print(df.columns)

# Séparer les données en ensembles d'entraînement et de test
X = df[df.columns]
exclude_columns = ['Sales', 'Order Item Quantity']
X = X.drop(columns=exclude_columns)
y_sales = df['Sales']
y_quantity = df['Order Item Quantity']

label_encoder = LabelEncoder()
categorical_columns = X.select_dtypes(include=['object']).columns

for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

X_train, X_test, y_sales_train, y_sales_test, y_quantity_train, y_quantity_test = train_test_split(
    X, y_sales, y_quantity, test_size=0.2, random_state=42
)

# Remplacer les valeurs manquantes par la moyenne
imputer = SimpleImputer()
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Initialiser le modèle de régression linéaire
model_sales = LinearRegression()
model_quantity = LinearRegression()

# Entraîner le modèle sur les données d'entraînement
model_sales.fit(X_train, y_sales_train)
model_quantity.fit(X_train, y_quantity_train)

# Prédire les valeurs sur l'ensemble de test
y_sales_pred = model_sales.predict(X_test)
y_quantity_pred = model_quantity.predict(X_test)

# Évaluer les performances du modèle en utilisant la MSE (Mean Squared Error)
mse_sales = mean_squared_error(y_sales_test, y_sales_pred)
mse_quantity = mean_squared_error(y_quantity_test, y_quantity_pred)

# Afficher les performances du modèle
print(f'Mean Squared Error (Sales): {mse_sales}')
print(f'Mean Squared Error (Quantity): {mse_quantity}')

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Lire le fichier CSV
file_name = 'DataCoSupplyChainDataset.csv'
df = pd.read_csv(file_name, encoding='latin-1')

# Afficher les premières lignes du dataframe
print(df.head())

# Afficher les types de données
print(df.dtypes)

# Afficher les valeurs uniques dans la colonne 'Type'
print(df['Type'].value_counts())

# Afficher la forme du dataframe
print(df.shape)

# Afficher le nombre de valeurs manquantes par colonne
print(df.isnull().sum())

# Supprimer certaines colonnes
df.drop(columns=['Product Description', 'Order Zipcode'], axis=1, inplace=True)

# Afficher des informations sur le dataframe
print(df.info())

# Afficher les noms des colonnes
print(df.columns)

# Afficher les valeurs uniques dans la colonne 'Late_delivery_risk'
print(df['Late_delivery_risk'].value_counts())

# Afficher les valeurs uniques dans la colonne 'Order Item Discount'
print(df['Order Item Discount'].value_counts())

# Afficher un graphique de régression
sns.regplot(x='Sales', y='Order Item Discount', data=df)
plt.show()

# Séparer les données en ensembles d'entraînement et de test
X = df.drop(columns=['Sales', 'Order Item Quantity'])
y_sales = df['Sales']
y_quantity = df['Order Item Quantity']

label_encoder = LabelEncoder()
categorical_columns = X.select_dtypes(include=['object']).columns

for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

X_train, X_test, y_sales_train, y_sales_test, y_quantity_train, y_quantity_test = train_test_split(
    X, y_sales, y_quantity, test_size=0.2, random_state=42
)

# Remplacer les valeurs manquantes par la moyenne
imputer = SimpleImputer()
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Initialiser le modèle d'arbre de décision pour les ventes
model_sales = DecisionTreeRegressor()
model_sales.fit(X_train, y_sales_train)

# Initialiser le modèle d'arbre de décision pour la quantité
model_quantity = DecisionTreeRegressor()
model_quantity.fit(X_train, y_quantity_train)

# Prédire les valeurs sur l'ensemble de test
y_sales_pred = model_sales.predict(X_test)
y_quantity_pred = model_quantity.predict(X_test)

# Évaluer les performances du modèle en utilisant la MSE (Mean Squared Error)
mse_sales = mean_squared_error(y_sales_test, y_sales_pred)
mse_quantity = mean_squared_error(y_quantity_test, y_quantity_pred)

# Afficher les performances du modèle
print(f'Mean Squared Error (Sales): {mean_squared_error(y_sales_test, y_sales_pred)}')
print(f'Accuracy (Sales): {accuracy_sales}')

print(f'Mean Squared Error (Quantity): {mean_squared_error(y_quantity_test, y_quantity_pred)}')
print(f'Accuracy (Quantity): {accuracy_quantity}')
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Lire le fichier CSV
file_name = 'DataCoSupplyChainDataset.csv'
df = pd.read_csv(file_name, encoding='latin-1')

# Supprimer certaines colonnes
df.drop(columns=['Product Description', 'Order Zipcode', 'Customer Email', 'Customer Fname', 'Customer Id',
                 'Customer Lname', 'Customer Password', 'Customer Street', 'Order City', 'Order Customer Id',
                 'order date (DateOrders)', 'Order Id', 'Order Item Cardprod Id'], axis=1, inplace=True)

# Séparer les données en ensembles d'entraînement et de test
X = df.drop(columns=['Sales', 'Order Item Quantity'])
y_sales = df['Sales']
y_quantity = df['Order Item Quantity']

# Convertir les valeurs de la cible en classes (exemple : [0, 1, 2])
bins = [0, 50, 100, 150, 200, 250, 300, 400, 500, float('inf')]
labels = list(range(len(bins) - 1))
y_sales_class = pd.cut(y_sales, bins=bins, labels=labels)

label_encoder = LabelEncoder()
categorical_columns = X.select_dtypes(include=['object']).columns

for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

X_train, X_test, y_sales_train, y_sales_test, y_quantity_train, y_quantity_test = train_test_split(
    X, y_sales_class, y_quantity, test_size=0.2, random_state=42
)

# Remplacer les valeurs manquantes par la moyenne
imputer = SimpleImputer()
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Initialiser le modèle Naive Bayes pour les ventes
model_sales = GaussianNB()
model_sales.fit(X_train, y_sales_train)

# Prédire les classes sur l'ensemble de test pour les ventes
y_sales_pred = model_sales.predict(X_test)

# Calculer l'accuracy pour les ventes
accuracy_sales = accuracy_score(y_sales_test, y_sales_pred)

# Initialiser le modèle Naive Bayes pour la quantité
model_quantity = GaussianNB()
model_quantity.fit(X_train, y_quantity_train)

# Prédire les valeurs sur l'ensemble de test pour la quantité
y_quantity_pred = model_quantity.predict(X_test)

# Calculer l'accuracy pour la quantité
accuracy_quantity = accuracy_score(y_quantity_test, y_quantity_pred)

# Afficher les performances du modèle
print(f'Mean Squared Error (Sales): {mean_squared_error(y_sales_test, y_sales_pred)}')
print(f'Accuracy (Sales): {accuracy_sales}')

print(f'Mean Squared Error (Quantity): {mean_squared_error(y_quantity_test, y_quantity_pred)}')
print(f'Accuracy (Quantity): {accuracy_quantity}')
"""
