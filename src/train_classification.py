import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "data/processed/processed_data.csv"
MODEL_PATH = "models/classification_model.pkl"


def train_classification_model():
    print("🔹 Loading processed data...")
    df = pd.read_csv(DATA_PATH)

    # Target
    y = df["conversion"]

    # Drop non-useful columns
    X = df.drop(
    columns=[
        "conversion",
        "session ID",
        "page",
        "order",
        "page 2 (clothing model)"
    ]
)

    # Encode categorical columns
    cat_cols = X.select_dtypes(include="object").columns
    print("🔹 Encoding categorical columns:", list(cat_cols))

    encoder = LabelEncoder()
    for col in cat_cols:
        X[col] = encoder.fit_transform(X[col])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🔹 Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {acc:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("✅ Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    train_classification_model()


