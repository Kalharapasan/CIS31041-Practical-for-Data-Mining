
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv('Bank1.csv')

# -------------------------------
# K-Means Clustering
# -------------------------------
# Select numeric columns
data = df.select_dtypes(include=[np.number])

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Elbow method to find optimal k
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.grid(True)
plt.show()

# Apply KMeans with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
df['Cluster'] = clusters

# Visualize clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=clusters, palette='Set2')
plt.title('Cluster Visualization')
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.show()

# Cluster centroids
centroids = kmeans.cluster_centers_
print("Cluster Centroids (scaled space):\n", centroids)

# Plot centroids
plt.figure(figsize=(8,6))
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters, cmap='viridis', label='Data')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.title('Cluster Plot with Centroids')
plt.legend()
plt.show()

# -------------------------------
# Decision Tree Classification
# -------------------------------
# Feature-target split
X = df.drop(['PEP'], axis=1)
y = df['PEP']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
