import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Upload data
data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Encoding the target 'Churn
label_encoder = LabelEncoder()
data['Churn'] = label_encoder.fit_transform(data['Churn'])

# Define numerical and categorical columns
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges'] 
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService'] 

# Handling Missing Data 
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')  # Converting invalid values
data.fillna(0, inplace=True)

# Data Preprocessing : Separate features from target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Pipeline : Create a preprocessor to manage the transformation pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Pipeline : Create the pipeline with the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Data Spliting : Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model pipeline 
model_pipeline.fit(X_train, y_train)

# Assessing the model
y_pred = model_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the complete pipeline
joblib.dump(model_pipeline, 'models/churn_model.pkl')
