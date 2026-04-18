# 📊 Pharmacy OTC Sales Prediction (Data Science Project)

---

## 📝 Project Overview

This project focuses on predicting **OTC pharmacy sales revenue** using historical transaction data.

Instead of just predicting sales, the goal is to:

> 🎯 **Forecast demand and support inventory optimization using data-driven insights**

This project simulates a real-world business scenario where pharmacies need to manage stock efficiently and maximize revenue.

---

## 🎯 Business Objective

* Forecast product demand
* Optimize inventory and reduce stockouts
* Identify key drivers of sales performance

---

## 📁 Dataset Description

The dataset contains pharmacy sales records with features such as:

* **Date** → Transaction date
* **Product** → Product category
* **Sales Person** → Sales representative
* **Boxes Shipped** → Quantity sold
* **Amount ($)** → Revenue generated (target variable)
* **Country** → Sales region

---

## ⚠️ Data Challenges

* Mixed categorical variables
* Sales variability across products and regions
* Right-skewed sales distribution
* Strong dependency between quantity and revenue

---

## 🔧 Project Workflow

### 1️⃣ Data Cleaning

* Removed duplicates
* Checked for missing values
* Ensured consistent data types

---

### 2️⃣ Exploratory Data Analysis (EDA)

Performed detailed analysis to understand patterns:

* 📊 Sales by Country
* 📊 Sales by Product
* 📊 Sales by Sales Person
* 📊 Sales Distribution
* 📊 Correlation Heatmap

📌 **Key Insights:**

* USA generates the highest revenue
* Certain products (e.g., Enzyme) dominate sales
* Sales performance varies across representatives
* Sales distribution is right-skewed
* Boxes shipped strongly correlates with revenue

---

### 3️⃣ Feature Engineering

Created business-driven features:

* **Revenue per Box** = Amount / Boxes Shipped
* **High Demand Flag** based on shipment volume

---

### 4️⃣ Data Preprocessing

* Applied **One-Hot Encoding** to categorical features
* Used pipeline for clean preprocessing + modeling

---

### 5️⃣ Model Building

Trained and compared:

* **Linear Regression (Baseline)**
* **Random Forest Regressor (Advanced Model)**

---

### 6️⃣ Model Evaluation

Metrics used:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## 📈 Final Model Performance

### 🔹 Linear Regression

* MAE: **53.27**
* RMSE: **66.77**
* R² Score: **0.73**

### 🔹 Random Forest

* MAE: **47.32**
* RMSE: **62.47**
* R² Score: **0.76**

---

## 🧠 Model Insights

* Random Forest outperforms Linear Regression
* Sales are strongly influenced by **shipment volume**
* Product type and region add additional variability
* Non-linear relationships exist in the data

---

## 💡 Business Impact

* Helps forecast demand accurately
* Enables better inventory planning
* Identifies high-performing products and regions
* Supports data-driven decision making

---

## 📊 Visualizations

* Sales by Country
* Sales by Product
* Sales by Sales Person
* Sales Distribution
* Correlation Heatmap
* True vs Predicted Plot

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn

---

## 🚀 Key Learnings

* Importance of EDA in understanding business data
* Handling skewed distributions in regression
* Feature engineering using domain knowledge
* Model comparison for better performance

---

## 📌 Conclusion

This project demonstrates how machine learning can be used to **predict sales and support business decisions** in a real-world pharmacy setting.

---

## 📎 How to Run

```bash
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
python model.py
```

