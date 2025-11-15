import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
df = pd.read_csv("customer.csv")
X = df[['Age', 'Total_Amount', 'Quantity', 'Session_Duration_Minutes', 'Pages_Viewed']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
plt.figure(figsize=(8,6))
sns.scatterplot(
    x='Total_Amount',
    y='Pages_Viewed',
    hue='Cluster',
    data=df,
    palette='Set2',
    s=100
)
plt.title("Customer Segments based on Spending and Browsing")
plt.xlabel("Total Amount Spent")
plt.ylabel("Pages Viewed")
plt.show()
cluster_summary = df.groupby('Cluster')[['Age', 'Total_Amount', 'Quantity']].mean()
print("\n--- Cluster Summary ---")
print(cluster_summary)
df.to_csv("segmented_customers.csv", index=False)
print("\n Segmentation completed! Results saved as 'segmented_customers.csv'")

