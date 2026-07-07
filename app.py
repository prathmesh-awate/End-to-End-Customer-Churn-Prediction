import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("logistic_regression.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")

st.title("📊 Customer Churn Prediction")
st.write("Enter the customer details below and click **Predict**.")

# -----------------------------
# Numerical Inputs
# -----------------------------
tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=850.0
)

# -----------------------------
# Categorical Inputs
# -----------------------------
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.selectbox(
    "Senior Citizen",
    ["Yes", "No"]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

phone = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):

    # Create DataFrame
    user = pd.DataFrame({
        "Tenure Months": [tenure],
        "Monthly Charges": [monthly],
        "Total Charges": [total],
        "Gender": [gender],
        "Senior Citizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "Phone Service": [phone],
        "Multiple Lines": [multiple],
        "Internet Service": [internet],
        "Online Security": [security],
        "Online Backup": [backup],
        "Device Protection": [device],
        "Tech Support": [support],
        "Streaming TV": [tv],
        "Streaming Movies": [movies],
        "Contract": [contract],
        "Paperless Billing": [paperless],
        "Payment Method": [payment]
    })

    # One-hot encode
    user = pd.get_dummies(user)

    # Match training columns
    user = user.reindex(columns=features, fill_value=0)

    # Scale features
    user_scaled = scaler.transform(user)

    # Prediction
    prediction = model.predict(user_scaled)[0]
    probability = model.predict_proba(user_scaled)[0][1]

    # Display Result
    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is unlikely to churn.")

    st.metric(
        label="Churn Probability",
        value=f"{probability:.2%}"
    )

    st.progress(float(probability))

    st.write("### Prediction Summary")

    result = pd.DataFrame({
        "Prediction": [
            "Churn" if prediction == 1 else "No Churn"
        ],
        "Probability": [
            f"{probability:.2%}"
        ]
    })

    st.dataframe(result, use_container_width=True)