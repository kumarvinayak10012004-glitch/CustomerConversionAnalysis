import pandas as pd
import os

RAW_PATH = "data/raw/clickstream+data+for+online+shopping/e-shop clothing 2008.csv"
PROCESSED_PATH = "data/processed/processed_data.csv"

def preprocess_data():
    print("🔹 Loading raw data...")
    df = pd.read_csv(RAW_PATH, sep=';')

    print("🔹 Shape before cleaning:", df.shape)

    # Drop missing
    df = df.dropna()

    # 🎯 CREATE TARGET COLUMN
    df["conversion"] = df["page"].apply(lambda x: 1 if x == 5 else 0)

    print("🔹 Conversion distribution:")
    print(df["conversion"].value_counts())

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("✅ Preprocessing done. File saved at:", PROCESSED_PATH)
    return df


if __name__ == "__main__":
    preprocess_data()


