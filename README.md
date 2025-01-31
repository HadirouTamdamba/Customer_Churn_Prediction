# Customer Churn Prediction (Predict behavior to retain customers)

## Problem Statement
Predict whether a customer will churn (leave) a service based on their usage patterns and demographics.

## Approach
1. **Exploratory Data Analysis (EDA)**: Visualize data distributions and correlations.
2. **Data Preprocessing**: Setting up a data pipeline, handle missing values, encode categorical variables, normalize data.
3. **Model Training**: Train a Random Forest classifier.
4. **Model Evaluation**: Evaluate the model using accuracy and classification report.
5. **Deployment**: Deploy the model using Flask and Azure Web Services.

## Results
Accuracy: 0.7785663591199432

Classification Report:
               precision    recall  f1-score   support

           0       0.82      0.89      0.86      1036 
           1       0.61      0.46      0.53       373 

    accuracy                           0.78      1409 
   macro avg       0.71      0.68      0.69      1409 
weighted avg       0.77      0.78      0.77      1409 










