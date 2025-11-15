import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", palette="muted")
df = pd.read_csv("retail_sales_dataset.csv")
print("===== First 5 Rows =====")
print(df.head())
print("\n===== Missing Values =====")
print(df.isnull().sum())
df = df.dropna(subset=['Date', 'Total Amount'])
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.drop_duplicates()
print("\n===== Descriptive Statistics =====")
print(df.describe())
print(f"\nMean Sales: {df['Total Amount'].mean():.2f}")
print(f"Median Sales: {df['Total Amount'].median():.2f}")
print(f"Mode Sales: {df['Total Amount'].mode()[0]:.2f}")
print(f"Standard Deviation: {df['Total Amount'].std():.2f}")
print("\n===== Time Series Analysis =====")
time_series = df.groupby('Date')['Total Amount'].sum()
plt.figure(figsize=(12,6))
plt.plot(time_series.index, time_series.values, marker='o', linewidth=2)
plt.title("Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.show()
if 'Customer ID' in df.columns and 'Product Category' in df.columns:
    print("\n===== Customer & Product Analysis =====")
    top_customers = df.groupby('Customer ID')['Total Amount'].sum().nlargest(10)
    top_customers.plot(kind='bar', figsize=(10,5), title="Top 10 Customers by Sales")
    plt.show()

    top_products = df.groupby('Product Category')['Total Amount'].sum().nlargest(10)
    top_products.plot(kind='bar', figsize=(10,5), title="Top 10 Product Categories by Sales")
    plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
print("\n===== Recommendations =====")
print("""
1. Focus marketing on top-selling product categories and high-value customers.
2. Offer promotions during low-sales periods identified in the time series.
3. Reduce stock or offer discounts on low-performing items.
4. Analyze customer segments for targeted marketing.
5. Improve data collection for missing or inconsistent entries.
""")


