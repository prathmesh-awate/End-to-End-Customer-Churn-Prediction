import shap
import joblib

# Load trained model
model = joblib.load("logistic_regression.pkl")

# Create SHAP explainer
explainer = shap.Explainer(model, X_train)

# Compute SHAP values
shap_values = explainer(X_test)

# Summary plot
shap.plots.beeswarm(shap_values)

# Waterfall plot for first prediction
shap.plots.waterfall(shap_values[0])