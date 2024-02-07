import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
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

X_train, X_test, y_sales_train, y_sales_test = train_test_split(
    X, y_sales_class, test_size=0.2, random_state=42
)

# Remplacer les valeurs manquantes par la moyenne
imputer = SimpleImputer()
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Initialiser le modèle Naive Bayes
model_sales = GaussianNB()
model_sales.fit(X_train, y_sales_train)

# Prédire les classes sur l'ensemble de test
y_sales_pred = model_sales.predict(X_test)

# Évaluer les performances du modèle en utilisant la MSE (Mean Squared Error)
mse_sales = mean_squared_error(y_sales_test, y_sales_pred)

# Afficher les performances du modèle
print(f'Mean Squared Error (Sales): {mse_sales}')
