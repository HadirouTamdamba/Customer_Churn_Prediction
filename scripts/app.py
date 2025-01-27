from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Upload the data
model = joblib.load('models/churn_model.pkl')    

# Columns used in the model
model_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                  'PhoneService', 'InternetService', 'MonthlyCharges', 'TotalCharges']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Recovering data sent from the frontend
    data = request.json.get('features', {})

    try:
        # Create a DataFrame from the data sent
        df = pd.DataFrame([data])

        # Check that all necessary columns are present
        for col in model_features:
            if col not in df.columns:
                return jsonify({'error': f'Missing feature: {col}'}), 400

        # Prediction
        prediction = model.predict(df)

        # Convert the prediction into an understandable format
        result = 'Churn' if prediction[0] == 1 else 'No Churn'
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
     
