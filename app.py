import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import shap
from src.preprocessing import feature_engineering, encode_features
from src.explain import ModelExplainer

# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# ==========================================
# Load Model
# ==========================================

model = joblib.load(MODEL_DIR / "gradient_boosting.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
feature_columns = joblib.load(MODEL_DIR / "features.pkl")
metrics = pd.read_csv(MODEL_DIR / "model_metrics.csv")


explainer = ModelExplainer(
    MODEL_DIR / "gradient_boosting.pkl",
    MODEL_DIR / "background.pkl"
)

# ==========================================
# Page
# ==========================================

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")

st.title("📊 Customer Churn Prediction")

st.write("Enter customer details below.")

# ==========================================
# Sidebar Metrics
# ==========================================

selected = metrics[metrics["Model"] == "Gradient Boost"].iloc[0]

st.sidebar.header("Model Performance")

st.sidebar.metric(
    "Accuracy",
    f"{selected['Accuracy']:.2%}"
)

st.sidebar.metric(
    "Precision",
    f"{selected['Precision']:.2%}"
)

st.sidebar.metric(
    "Recall",
    f"{selected['Recall']:.2%}"
)

st.sidebar.metric(
    "F1 Score",
    f"{selected['F1-Score']:.2%}"
)

st.sidebar.metric(
    "ROC AUC",
    f"{selected['AUC Score']:.2%}"
)

# ==========================================
# Inputs
# ==========================================

tenure = st.number_input(
    "Tenure Months",
    0,
    100,
    12
)

monthly = st.number_input(
    "Monthly Charges",
    0.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    0.0,
    value=850.0
)

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
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
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

# ==========================================
# Prediction
# ==========================================

if st.button("Predict Churn"):

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

    # Same feature engineering as training
    user = feature_engineering(user)

    # Same encoding as training
    user = encode_features(user)

    # Match training columns
    user = user.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale
    user_scaled = scaler.transform(user)

    user_scaled = pd.DataFrame(
        user_scaled,
        columns=feature_columns
    )

    # Predict
    prediction = model.predict(user_scaled)[0]

    probability = model.predict_proba(user_scaled)[0][1]
    # SHAP explanation
    shap_values = explainer.explain(user_scaled)

    st.divider()

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is unlikely to churn.")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

    st.progress(float(probability))

    st.subheader("Prediction Summary")

    st.dataframe(
        pd.DataFrame({
            "Prediction": [
                "Churn" if prediction == 1 else "No Churn"
            ],
            "Probability": [
                f"{probability:.2%}"
            ]
        }),
        use_container_width=True
    )
    st.divider()
    st.subheader("📋 Top Contributing Features")

    # Create DataFrame of SHAP values
    shap_df = pd.DataFrame({
        "Feature": feature_columns,
        "SHAP Value": shap_values.values[0]
    })

    # Keep only the most influential features
    top_features = (
        shap_df.reindex(
            shap_df["SHAP Value"].abs().sort_values(ascending=False).index
        )
        .head(10)
    )

    # Show table
    st.dataframe(
        top_features,
        use_container_width=True,
        hide_index=True
    )
    st.subheader("📝 Model Explanation")

    explanations = []

    for _, row in top_features.head(5).iterrows():

        feature = row["Feature"].replace("_", " ")
        value = row["SHAP Value"]

        if value > 0:
            explanations.append(
                f"• **{feature}** increased the predicted likelihood of churn."
            )
        else:
            explanations.append(
                f"• **{feature}** decreased the predicted likelihood of churn."
            )

    for text in explanations:
        st.markdown(text)