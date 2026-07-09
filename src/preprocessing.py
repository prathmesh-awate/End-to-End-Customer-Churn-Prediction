import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ==========================================
# Load Data
# ==========================================

def load_data():
    df = pd.read_excel(
        "/Users/prath/Documents/End-to-End-Customer-Churn-Prediction/data/raw/Telco_customer_churn.xlsx"
    )
    return df


# ==========================================
# Data Cleaning
# ==========================================

def clean_data(df):

    columns_to_drop = [
        "CustomerID",
        "Lat Long",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Latitude",
        "Longitude",
        "Churn Label",
        "Churn Score",
        "Churn Reason",
        "CLTV"
    ]

    df = df.drop(columns=columns_to_drop)

    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce"
    )

    df = df.dropna()

    return df


# ==========================================
# Feature Engineering
# ==========================================

def feature_engineering(df):

    # Protection Services
    protection_cols = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support"
    ]

    df["Protection Services"] = (
        (df[protection_cols] == "Yes").sum(axis=1)
    )

    # Entertainment Services
    entertainment_cols = [
        "Streaming TV",
        "Streaming Movies"
    ]

    df["Entertainment Services"] = (
        (df[entertainment_cols] == "Yes").sum(axis=1)
    )

    # Tenure Groups
    df["Tenure Group"] = pd.cut(
        df["Tenure Months"],
        bins=[0, 12, 24, 48, float("inf")],
        labels=[
            "New",
            "Regular",
            "Loyal",
            "Very Loyal"
        ],
        include_lowest=True
    )

    # Contract Commitment
    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }

    df["Contract Commitment"] = df["Contract"].map(contract_map)

    return df


# ==========================================
# Encoding
# ==========================================

def encode_features(df):

    df = pd.get_dummies(df, dtype=int)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


# ==========================================
# Train Test Split + SMOTE
# ==========================================

def split_data(df):

    X = df.drop("Churn Value", axis=1)
    y = df["Churn Value"]

    # Save feature names for Streamlit
    joblib.dump(
        X.columns.tolist(),
        "/Users/prath/Documents/End-to-End-Customer-Churn-Prediction/models/features.pkl"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Apply SMOTE ONLY on training data
    smote = SMOTE(random_state=42)

    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )

    return X_train, X_test, y_train, y_test


# ==========================================
# Scaling
# ==========================================

def scale_data(X_train, X_test):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(
        scaler,
        "/Users/prath/Documents/End-to-End-Customer-Churn-Prediction/models/scaler.pkl"
    )

    return X_train, X_test


# ==========================================
# Complete Pipeline
# ==========================================

def preprocess_data():

    df = load_data()

    df = clean_data(df)

    df = feature_engineering(df)

    df = encode_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test = scale_data(
        X_train,
        X_test
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )