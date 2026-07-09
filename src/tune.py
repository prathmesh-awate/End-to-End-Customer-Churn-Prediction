import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import GridSearchCV

from preprocessing import preprocess_data


def tune_model(X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV.
    """

    param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3],
    "min_samples_split": [2],
    "min_samples_leaf": [1]
    }

    model = GradientBoostingClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross Validation F1:")
    print(grid_search.best_score_)

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the tuned model.
    """

    y_pred = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def save_model(model):
    """
    Save tuned model.
    """

    joblib.dump(
        model,
        "/Users/prath/Documents/End-to-End-Customer-Churn-Prediction/models/gradient_boosting.pkl"
    )

    print("\nModel saved successfully.")


def main():

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = preprocess_data()

    best_model = tune_model(X_train, y_train)

    evaluate_model(best_model, X_test, y_test)

    save_model(best_model)


if __name__ == "__main__":
    main()