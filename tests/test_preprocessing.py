from src.preprocessing import load_data, clean_data, feature_engineering, encode_features,  split_data, preprocess_data

def test_load_data():
    df = load_data()

    assert not df.empty


def test_clean_data():
    df = load_data()
    df = clean_data(df)

    assert "CustomerID" not in df.columns
    assert "CLTV" not in df.columns
    assert df["Total Charges"].isna().sum() == 0

def test_feature_engineering():
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)

    assert "Protection Services" in df.columns
    assert "Entertainment Services" in df.columns
    assert "Tenure Group" in df.columns
    assert "Contract Commitment" in df.columns

def test_encode_features():
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_features(df)

    assert df.select_dtypes(include="object").empty

def test_split_data():
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_scale_data():
    X_train, X_test, y_train, y_test = preprocess_data()

    assert X_train.shape[1] == X_test.shape[1]