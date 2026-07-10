import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from src.preprocessing import preprocess_data
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

def train_model(model, X_train, y_train): 
    """
    Train the Logistic Regression model.
    """

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    classification= classification_report(y_test, y_pred, output_dict=True)

    confusion = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    return accuracy, classification, confusion, auc


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

    models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42), 
    "Gradient Boost": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=42), 
    "Extreme Gradient Boosting": xgb.XGBClassifier(learning_rate= 0.01,max_depth = 3,n_estimators = 1000)
    }
    results = []

    for name, model in models.items():

        print(f"\n{'=' * 50}")
        print(f"Training {name}")
        print(f"{'=' * 50}")

        trained_model = train_model(model, X_train, y_train)

        accuracy, classification, confusion, auc= evaluate_model(trained_model, X_test, y_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy, 
            "Precision": classification["1"]["precision"],
            "Recall": classification["1"]["recall"],
            "F1-Score": classification["1"]["f1-score"],
            "Confusion Matrix": confusion, 
            "AUC Score": auc
        })
        if name == "Logistic Regression":
            save_model(trained_model)
    results_df = pd.DataFrame(results)
    
    print("\nModel Comparison")
    print(results_df)
    """The comparison shows that Gradient Boosting achieved the best overall performance across all evaluation metrics. 
    It produced the highest accuracy (80.70%), precision (66.35%), recall (55.35%), and F1-score (60.35%), 
    indicating a better balance between correctly identifying customers likely to churn and minimizing incorrect predictions."""

if __name__ == "__main__":
    main()