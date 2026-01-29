import pandas as pd

# 1️⃣ Load raw data
df = pd.read_csv("data/raw/e-shop clothing 2008.csv")

# 2️⃣ Target variable (Conversion)
# Assumption: PAGE >= 4 → Converted
df['converted'] = (df['PAGE'] >= 4).astype(int)

# 3️⃣ Feature Engineering
df['session_clicks'] = df.groupby('SESSION ID')['ORDER'].transform('count')
df['avg_price_session'] = df.groupby('SESSION ID')['PRICE'].transform('mean')

# 4️⃣ Save processed data
df.to_csv("data/processed/processed_data.csv", index=False)

print("✅ processed_data.csv created successfully")

