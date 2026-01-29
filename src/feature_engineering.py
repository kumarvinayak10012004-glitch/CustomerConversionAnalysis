import pandas as pd
from src.data_preprocessing import handle_missing_values
from src.feature_engineering import add_session_features

# Load raw dataset
df = pd.read_csv("data/raw/train.csv")

print("Initial dataset shape:", df.shape)
print(df.head())
