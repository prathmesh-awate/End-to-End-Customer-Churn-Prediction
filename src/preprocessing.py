# Data Cleaning

from curses import raw
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data():
    df = pd.read_excel("/Users/prath/Documents/End-to-End-Customer-Churn-Prediction/data/raw/Telco_customer_churn.xlsx")
    return df
    
def clean_data(df):
    """
    Clean the data by removing missing values, duplicates, and outliers.
    """
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
    "CLTV" ]

    df = df.drop(columns=columns_to_drop)
    return df

def encode_features(df):
    # One hot encoding to convert categorical data to numerical data
    df = pd.get_dummies(df, dtype=int)
    bool_cols = df.select_dtypes(include="bool").columns    
    df[bool_cols] = df[bool_cols].astype(int)
    X = df.drop(["Churn Value"], axis=1) 
    y = df["Churn Value"] # Target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )   

    feature_columns = X_train.columns.tolist()

    joblib.dump(feature_columns, "/Users/prath/Documents/End-to-End-Customer-Churn-Prediction/models/features.pkl")
    return df

def split_data(df):
    # Splitting features and target

    X = df.drop(["Churn Value"], axis=1) # Input dataset(only features) shouldn't contain the output
    y = df["Churn Value"] # Target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )   
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    # Scaling numerical data to make them unitless

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, "/Users/prath/Documents/End-to-End-Customer-Churn-Prediction/models/scaler.pkl")
    return X_train, X_test

def preprocess_data():
    df = load_data()
    df = clean_data(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = scale_data(X_train, X_test)

    return X_train, X_test, y_train, y_test