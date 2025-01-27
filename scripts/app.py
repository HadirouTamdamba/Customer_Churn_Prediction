from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Charger le modèle
model = joblib.load('models/churn_model.pkl')    

# Colonnes utilisées par le modèle
model_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                  'PhoneService', 'InternetService', 'MonthlyCharges', 'TotalCharges']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données envoyées depuis le frontend
    data = request.json.get('features', {})

    try:
        # Créer un DataFrame à partir des données envoyées
        df = pd.DataFrame([data])

        # Vérifier que toutes les colonnes nécessaires sont présentes
        for col in model_features:
            if col not in df.columns:
                return jsonify({'error': f'Missing feature: {col}'}), 400

        # Faire la prédiction
        prediction = model.predict(df)

        # Convertir la prédiction en un format compréhensible
        result = 'Churn' if prediction[0] == 1 else 'No Churn'
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
