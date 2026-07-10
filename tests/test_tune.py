import os

from src.preprocessing import preprocess_data
from src.tune import tune_model, save_model


def test_tune_pipeline():

    # Load data
    X_train, X_test, y_train, y_test = preprocess_data()

    # Tune model
    best_model = tune_model(X_train, y_train)

    # Check model was created
    assert best_model is not None

    # Make predictions
    predictions = best_model.predict(X_test)

    # Check predictions
    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})

    # Save model
    save_model(best_model)

    # Check model file exists
    assert os.path.exists(
        "/Users/prath/Documents/End-to-End-Customer-Churn-Prediction/models/gradient_boosting.pkl"
    )