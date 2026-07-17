import shap
import joblib


class ModelExplainer:
    def __init__(self, model_path, background_path):
        self.model = joblib.load(model_path)
        background = joblib.load(background_path)
        self.explainer = shap.Explainer(self.model, background)

    def explain(self, data):
        return self.explainer(data)