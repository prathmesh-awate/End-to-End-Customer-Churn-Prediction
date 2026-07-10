import joblib

model = joblib.load("models/gradient_boosting.pkl")

print(type(model))
print(model)