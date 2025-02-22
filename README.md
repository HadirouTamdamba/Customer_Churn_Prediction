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
Accuracy: 0.78
Precision : 0.61
F1-score : 0.53

## How to Run
1. Clone the repository.
2. Install dependencies: "pip install -r app/requirements.txt".
3. Run the Flask app: "python app/app.py".
4. Access the app at : **http://127.0.0.1:5000/**.

## Future Work
- Improve model performance using hyperparameter tuning.
- Add more features to the dataset.










