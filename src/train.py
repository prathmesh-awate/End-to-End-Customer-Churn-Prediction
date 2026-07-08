import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



from preprocessing import preprocess_data

def train_model(X_train, y_train): 
    """
    Train the Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    """
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def save_model(model):
    """
    Save trained model.
    """
    joblib.dump(model, "/Users/prath/Documents/End-to-End-Customer-Churn-Prediction/models/logistic_regression.pkl")
    print("Model saved successfully.")


def main():
    """
    Complete training pipeline.
    """

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = preprocess_data()

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)


if __name__ == "__main__":
    main()