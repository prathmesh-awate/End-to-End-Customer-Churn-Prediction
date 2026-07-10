import os
from sklearn.linear_model import LogisticRegression

from src.preprocessing import preprocess_data
from src.train import train_model, evaluate_model, save_model

def test_train_pipeline():

    # Load data
    X_train, X_test, y_train, y_test = preprocess_data()

    # Create model
    model = LogisticRegression(random_state=42, max_iter=1000)

    # Train model
    trained_model = train_model(model, X_train, y_train)

    # Check model was trained
    assert trained_model is not None

    # Evaluate model
    accuracy, classification, confusion, auc = evaluate_model(
        trained_model,
        X_test,
        y_test
    )

    # Check evaluation metrics
    assert 0 <= accuracy <= 1
    assert 0 <= auc <= 1
    assert confusion.shape == (2, 2)
    assert "1" in classification

    # Save model
    save_model(trained_model)

    # Check model file exists
    assert os.path.exists(
        "/Users/prath/Documents/End-to-End-Customer-Churn-Prediction/models/logistic_regression.pkl"
    )