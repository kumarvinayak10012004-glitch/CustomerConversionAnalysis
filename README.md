# 🛒 Customer Conversion Analysis Using Clickstream Data

## 📌 Domain
E-commerce | Retail Analytics | Machine Learning | Customer Behavior Analysis

---

## 📖 Project Overview
This project focuses on analyzing customer clickstream data to understand online shopping behavior and improve conversion rates.  
An end-to-end Machine Learning solution has been developed that performs:

- Conversion Prediction (Classification)
- Revenue Estimation (Regression)
- Customer Segmentation (Clustering)

The final solution is deployed as an interactive Streamlit web application that allows businesses to analyze customer behavior and generate real-time predictions.

---

## 🎯 Problem Statement
The goal is to build an intelligent system that analyzes browsing sessions and predicts:

- Whether a customer will complete a purchase
- Estimated revenue from each user
- Behavioral customer segments for personalized marketing

This system helps e-commerce companies enhance customer engagement, increase conversions, and optimize revenue strategies.

---

## 💼 Business Use Cases
- 🎯 Customer Conversion Prediction – Identify potential buyers
- 💰 Revenue Forecasting – Predict customer spending
- 👥 Customer Segmentation – Behavioral grouping for marketing
- 🔁 Churn Reduction – Detect users likely to abandon carts
- 🛍️ Personalized Recommendations – Suggest products based on browsing patterns

---

## 🧠 Skills Demonstrated
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Classification, Regression & Clustering Models
- Hyperparameter Tuning
- Model Evaluation Techniques
- Machine Learning Pipelines
- Streamlit Web Application Development
- Model Deployment

---

## 🏗️ Project Workflow

### 📊 1. Data Preprocessing
- Handling missing values using mean, median, and mode
- Encoding categorical variables using Label Encoding & One-Hot Encoding
- Feature scaling using StandardScaler & MinMaxScaler
- Dataset split into training and testing sets

Datasets:
- Train.csv
- Test.csv

---

### 📈 2. Exploratory Data Analysis
- Distribution analysis using histograms and bar charts
- Session behavior analysis
- Correlation heatmaps
- Time-based browsing analysis
- Page view and bounce rate analysis

---

### 🧩 3. Feature Engineering
- Session duration and click counts
- Page visit sequences
- Bounce rate & exit rate calculation
- Customer browsing patterns
- Behavioral metrics

---

### ⚖️ 4. Data Balancing (Classification)
- Target distribution analysis
- SMOTE for oversampling
- Random undersampling
- Class weight balancing

---

### 🤖 5. Model Development

#### Classification Models
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Neural Networks

#### Regression Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Gradient Boosting Regressor

#### Clustering Models
- K-Means
- DBSCAN
- Hierarchical Clustering

#### Pipeline Development
- Data preprocessing
- Feature scaling
- Model training
- Hyperparameter tuning
- Model evaluation

---

### 📏 6. Model Evaluation

#### Classification Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

#### Regression Metrics
- MAE
- MSE
- RMSE
- R² Score

#### Clustering Metrics
- Silhouette Score
- Davies-Bouldin Index
- Within Cluster Sum of Squares

---

### 🌐 7. Streamlit Application
Features:
- Upload CSV files or input values manually
- Predict customer conversion
- Estimate revenue potential
- Display customer clusters
- Interactive visualizations and analytics

---

## 📊 Results
- Accurate customer conversion prediction
- Reliable revenue estimation models
- Meaningful behavioral customer segmentation
- Fully functional Streamlit web application

---

## 🛠️ Technology Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- TensorFlow
- Streamlit
- Git & GitHub

---

## 📂 Dataset Information

Source:
UCI Machine Learning Repository – Clickstream Dataset

Key Variables:
- Year, Month, Day
- Session ID
- Country
- Page Category
- Clothing Model
- Product Color
- Photo Location
- Model Photography Type
- Product Price
- Page Number
- Click Order Sequence

Dataset Characteristics:
- Sequential click data
- Categorical and numerical features
- Behavioral browsing information
- Session-level tracking

---

## 🚀 Installation & Setup

### Clone Repository
```bash
git clone <your-repo-link>
cd project-folder
Install Dependencies
pip install -r requirements.txt
Run Application
streamlit run app.py
📁 Project Structure
CustomerConversionProject/
│
├── data/
├── notebooks/
├── models/
├── src/
├── artifacts/
├── app.py
├── requirements.txt
└── README.md

👤 Author
Vinayak Kumar
Data Science | Machine Learning | Computer Vision

⭐ If you like this project, give it a star!
