import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Charger les données
data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Encodage de la cible 'Churn'
label_encoder = LabelEncoder()
data['Churn'] = label_encoder.fit_transform(data['Churn'])

# Définir les colonnes numériques et catégoriques
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']  # Exemple de colonnes numériques
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService']  # Exemple de colonnes catégoriques

# Remplir les valeurs manquantes
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')  # Conversion des valeurs invalides
data.fillna(0, inplace=True)

# Séparer les caractéristiques et la cible
X = data.drop('Churn', axis=1)
y = data['Churn']

# Créer un préprocesseur pour gérer le pipeline de transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Créer le pipeline avec le modèle
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le pipeline
model_pipeline.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Sauvegarder le pipeline complet
joblib.dump(model_pipeline, 'models/churn_model.pkl')
