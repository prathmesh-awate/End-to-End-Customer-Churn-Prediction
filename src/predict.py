import joblib
import pandas as pd

# Load trained artifacts
model = joblib.load("/Users/prath/Documents/End-to-End-Customer-Churn-Prediction/models/logistic_regression.pkl")
scaler = joblib.load("/Users/prath/Documents/End-to-End-Customer-Churn-Prediction/models/scaler.pkl")
feature_columns = joblib.load("/Users/prath/Documents/End-to-End-Customer-Churn-Prediction/models/features.pkl")


def predict(user_input):
    """
    Predict customer churn.

    Parameters
    ----------
    user_input : dict
        Dictionary containing user input from Streamlit.

    Returns
    -------
    prediction : int
        0 = No Churn
        1 = Churn

    probability : float
        Probability of churn.
    """

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([user_input])

    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df, dtype=int)

    # Add missing columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep the same column order as training
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scale the data
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Prediction probability
    probability = model.predict_proba(input_scaled)[0][1]

    return prediction, probability