
# ==============================
# 📦 STEP 0: Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================
# 📂 STEP 1: Load Dataset
# ==============================
df = pd.read_csv("pharmacy_otc_sales_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# ==============================
# 🧹 STEP 2: Data Cleaning
# ==============================

# Remove duplicates
df.drop_duplicates(inplace=True)

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Fill missing values if needed (example)
df.fillna(0, inplace=True)


# ==============================
# 📊 STEP 3: Exploratory Data Analysis (EDA)
# ==============================

# Total Sales by Country
plt.figure(figsize=(8,5))
sns.barplot(x='Country', y='Amount ($)', data=df, estimator=sum)
plt.title('Total Sales by Country')
plt.xticks(rotation=45)
plt.show()

# Total Sales by Product
plt.figure(figsize=(8,5))
sns.barplot(x='Product', y='Amount ($)', data=df, estimator=sum)
plt.title('Total Sales by Product')
plt.xticks(rotation=45)
plt.show()

# Sales by Sales Person
plt.figure(figsize=(8,5))
sns.barplot(x='Sales Person', y='Amount ($)', data=df, estimator=sum)
plt.title('Sales by Sales Person')
plt.xticks(rotation=45)
plt.show()

# Sales Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Amount ($)'], bins=30, kde=True)
plt.title("Sales Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues")
plt.title('Correlation Heatmap')
plt.show()


# ==============================
# ⚙️ STEP 4: Feature Engineering
# ==============================

# Revenue per box (important business feature)
df['Revenue_per_Box'] = df['Amount ($)'] / df['Boxes Shipped']

# High demand flag (optional classification insight)
df['High_Demand'] = df['Boxes Shipped'].apply(
    lambda x: 1 if x > df['Boxes Shipped'].median() else 0
)


# ==============================
# 🎯 STEP 5: Define Features & Target
# ==============================

X = df[['Product', 'Sales Person', 'Boxes Shipped', 'Country']]
y = df['Amount ($)']


# ==============================
# 🔤 STEP 6: Preprocessing (Encoding)
# ==============================

categorical_features = ['Product', 'Sales Person', 'Country']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Keep numeric features
)


# ==============================
# 🧪 STEP 7: Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 🤖 STEP 8: Model Building
# ==============================

# 🔹 Linear Regression (Baseline Model)
lr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# 🔹 Random Forest (Advanced Model)
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=200, random_state=42))
])


# ==============================
# 🚀 STEP 9: Train Models
# ==============================

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)


# ==============================
# 📈 STEP 10: Model Evaluation
# ==============================

def evaluate_model(name, y_true, y_pred):
    print(f"\n🔹 {name} Performance:")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2 Score:", r2_score(y_true, y_pred))


# Predictions
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Evaluate
evaluate_model("Linear Regression", y_test, lr_pred)
evaluate_model("Random Forest", y_test, rf_pred)


# ==============================
# 📊 STEP 11: Visualization (True vs Predicted)
# ==============================

plt.figure(figsize=(6,6))
plt.scatter(y_test, lr_pred, alpha=0.6)
plt.xlabel("True Sales")
plt.ylabel("Predicted Sales")
plt.title("Linear Regression: True vs Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()


# ==============================
# 💼 STEP 12: Business Insights (WRITE IN NOTEBOOK)
# ==============================

# Example Insights:
# - High shipment volume strongly drives revenue
# - Certain products contribute most to total sales
# - Sales vary across countries and sales representatives
# - Useful for inventory planning and demand forecasting

# ==============================
# ✅ PROJECT COMPLETE
# ==============================
