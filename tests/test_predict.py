from src.predict import predict

def test_predict():

    user_input = {
        "Tenure Months": 12,
        "Monthly Charges": 70.0,
        "Total Charges": 850.0,
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check"
    }

    prediction, probability = predict(user_input)

    assert prediction in [0, 1]
    assert 0 <= probability <= 1





