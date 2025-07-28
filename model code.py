import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load Dataset
df = pd.read_csv("pharmacy_otc_sales_data.csv")

# Step 2: Basic EDA
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

# Plot total sales by country
plt.figure(figsize=(8,5))
sns.barplot(x='Country', y='Amount ($)', data=df, estimator=sum)
plt.title('Total Sales by Country')
plt.xticks(rotation=45)
plt.show()

# Correlation Heatmap (Numeric Features)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues")
plt.title('Correlation Heatmap')
plt.show()

# Step 3: Define Features and Target
# Target: Amount ($)
X = df[['Product', 'Sales Person', 'Boxes Shipped', 'Country']]
y = df['Amount ($)']

# Step 4: Preprocessing - OneHotEncode categorical features
categorical_features = ['Product', 'Sales Person', 'Country']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Keep 'Boxes Shipped' as is
)

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build pipeline with preprocessing and model
linear_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Step 7: Train the model
linear_model.fit(X_train, y_train)

# Step 8: Predict on test set
y_pred = linear_model.predict(X_test)

# Step 9: Evaluate the model
print("Linear Regression Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2 Score:", r2_score(y_test, y_pred))

# Step 10: Plot true vs predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("True Amount ($)")
plt.ylabel("Predicted Amount ($)")
plt.title("True vs Predicted Sales Amount")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()
