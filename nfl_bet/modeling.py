from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd


def build_preprocessor(numerical_features, categorical_features):
    return ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(), categorical_features),
    ])


def train_model(
    df: pd.DataFrame,
    features,
    categorical_features=None,
    target="dog_win",
    test_size=0.2,
    random_state=42,
    model_type: str = "classification",
):
    if categorical_features is None:
        categorical_features = []
    numerical_features = [f for f in features if f not in categorical_features]
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    X = df[features]
    y = df[target]
    stratify = y if model_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
    ])
    X_train_trans = pipeline.fit_transform(X_train)
    X_test_trans = pipeline.transform(X_test)
    if model_type == "regression":
        model = LinearRegression()
    else:
        model = LogisticRegression(max_iter=1000)
    model.fit(X_train_trans, y_train)
    return model, pipeline, (X_test_trans, y_test)
