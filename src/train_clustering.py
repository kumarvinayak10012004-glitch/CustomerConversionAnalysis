# src/train_clustering.py

import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans

# ✅ Correct import of preprocessing function
from src.data_preprocessing import preprocess_data

# -----------------------------
# Step 1: Load and preprocess data
# -----------------------------
df = preprocess_data()  # Returns the processed DataFrame

# -----------------------------
# Step 2: Prepare data for clustering
# -----------------------------
# Drop target columns if exist
cols_to_drop = []
for col in ['price', 'price 2']:
    if col in df.columns:
        cols_to_drop.append(col)

X_cluster = df.drop(columns=cols_to_drop) if cols_to_drop else df.copy()

# Convert categorical columns to numeric
X_cluster = pd.get_dummies(X_cluster, drop_first=True)

# -----------------------------
# Step 3: Train clustering model
# -----------------------------
# Example: 3 clusters (you can change n_clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_cluster)

# -----------------------------
# Step 4: Save clustering model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(kmeans, "models/clustering_model.pkl")
print("✅ Clustering model trained and saved at: models/clustering_model.pkl")

