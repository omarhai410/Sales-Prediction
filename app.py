from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


app = Flask(__name__)

# Charger les données d'entraînement depuis un fichier Excel
data_train_linear = pd.read_excel('data_no_duplicates.xlsx')
data_train_rf = pd.read_excel('data_no_duplicates1.xlsx')

# Diviser le dataset d'entraînement pour le modèle de régression linéaire
X_train_linear = data_train_linear.drop('Sales', axis=1)
y_train_linear = data_train_linear['Sales']

# Initialiser le modèle de régression linéaire
linear_model = LinearRegression()

# Entraîner le modèle sur l'ensemble d'entraînement pour le modèle de régression linéaire
linear_model.fit(X_train_linear, y_train_linear)

# Diviser le dataset d'entraînement pour le modèle Random Forest
X_train_rf = data_train_rf.drop('Order Item Quantity', axis=1)
y_train_rf = data_train_rf['Order Item Quantity']

# Initialiser le modèle Random Forest (corriger LinearRegression par RandomForestRegressor)
random_forest_model = RandomForestRegressor()

# Entraîner le modèle sur l'ensemble d'entraînement pour le modèle Random Forest
random_forest_model.fit(X_train_rf, y_train_rf)


@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict_linear', methods=['POST'])
def predict_linear():
    # Récupérer les données de test depuis la requête POST
    user_input = request.json

    # Convertir les données de test en DataFrame
    test_data_input = pd.DataFrame([user_input])

    # Prédire les valeurs sur l'ensemble de test fourni par l'utilisateur pour le modèle de régression linéaire
    y_pred = linear_model.predict(test_data_input)

    # Ajouter les prédictions au DataFrame
    test_data_input['Sales_Predicted'] = y_pred

    # Convertir le DataFrame en dictionnaire JSON
    result = test_data_input.to_dict(orient='records')[0]

    return jsonify(result)

@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    try:
        # Récupérer les données de test depuis la requête POST
        user_input = request.json

        # Convertir les données de test en DataFrame
        test_data_input = pd.DataFrame([user_input])

        # Prédire les valeurs sur l'ensemble de test fourni par l'utilisateur pour le modèle Random Forest
        y_pred = random_forest_model.predict(test_data_input)

        # Ajouter les prédictions au DataFrame
        test_data_input['Quantity_Predicted'] = y_pred  # Change this line

        # Convertir le DataFrame en dictionnaire JSON
        result = test_data_input.to_dict(orient='records')[0]

        return jsonify(result)
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': 'An internal server error occurred'}), 500

@app.route('/menu')
def menu():
    return render_template('menu.html')

if __name__ == '__main__':
    app.run(debug=True)
