# app/app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Customer Conversion Analysis", layout="wide")
st.title("Customer Conversion Analysis App")

# -----------------------------
# Step 1: Load models
# -----------------------------
@st.cache_resource
def load_models():
    clf = joblib.load("models/classification_model.pkl")
    reg = joblib.load("models/regression_model.pkl")
    clu = joblib.load("models/clustering_model.pkl")
    return clf, reg, clu

clf, reg, clu = load_models()
st.success("✅ Models loaded successfully!")

# -----------------------------
# Step 2: File uploader
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV file for prediction", type="csv")

if uploaded_file is not None:
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file, sep=';')
    except Exception:
        df = pd.read_csv(uploaded_file)
    
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Step 3: Preprocess data
    # -----------------------------
    # Drop target / unused columns
    drop_cols = ['price', 'price 2', 'conversion']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # -----------------------------
    # Handle high-cardinality columns with LabelEncoder
    # -----------------------------
    label_cols = ['session ID', 'page 2 (clothing model)']
    for col in label_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # -----------------------------
    # One-hot encode remaining categorical columns
    # -----------------------------
    dummy_cols = ['colour', 'country', 'day', 'location', 'model photography', 'page 1 (main category)']
    df = pd.get_dummies(df, columns=[c for c in dummy_cols if c in df.columns], drop_first=True)

    # -----------------------------
    # Step 4: Classification prediction
    # -----------------------------
    for col in clf.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df_class = df[clf.feature_names_in_]
    df['Converted'] = clf.predict(df_class)

    # -----------------------------
    # Step 5: Regression prediction
    # -----------------------------
    X_reg = df.copy()
    for col in ['Converted', 'Predicted_Price', 'Cluster_Label']:
        if col in X_reg.columns:
            X_reg = X_reg.drop(columns=[col])
    for col in reg.feature_names_in_:
        if col not in X_reg.columns:
            X_reg[col] = 0
    X_reg = X_reg[reg.feature_names_in_]
    df['Predicted_Price'] = reg.predict(X_reg)

    # -----------------------------
    # Step 6: Clustering prediction
    # -----------------------------
    X_clu = df.copy()
    for col in ['Converted', 'Predicted_Price', 'Cluster_Label']:
        if col in X_clu.columns:
            X_clu = X_clu.drop(columns=[col])
    for col in clu.feature_names_in_:
        if col not in X_clu.columns:
            X_clu[col] = 0
    X_clu = X_clu[clu.feature_names_in_]
    df['Cluster_Label'] = clu.predict(X_clu)

    # -----------------------------
    # Step 7: Show results
    # -----------------------------
    st.subheader("Predictions")
    st.dataframe(df.head())

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv'
    )



