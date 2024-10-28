import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, num_clusters, threshold):
        self.num_clusters = num_clusters
        self.threshold = threshold
        self.centroids = None
        self.data = []

    def _euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def fit(self, x, epochs):
        # Randomly initialize centroids
        indices = np.random.choice(len(x), self.num_clusters, replace=False)
        self.centroids = x[indices]

        for i in range(epochs):
            self.data = [[] for _ in range(self.num_clusters)]

            # Assignment step
            for datapoint in x:
                distances = [self._euclidean_distance(datapoint, self.centroids[j]) for j in range(self.num_clusters)]
                min_idx = np.argmin(distances)
                self.data[min_idx].append(datapoint)

            # Update step
            new_centroids = []
            for j in range(self.num_clusters):
                if len(self.data[j]) == 0:
                    new_centroids.append(self.centroids[j])  # Keep the old centroid if no points assigned
                else:
                    new_centroids.append(np.mean(self.data[j], axis=0))  # Calculate the new centroid

            new_centroids = np.array(new_centroids)

            # Check for convergence
            if np.all(self.centroids == new_centroids):
                print("No change in means at epoch =", i + 1)
                break

            self.centroids = new_centroids

            print(f"Epoch {i + 1}: Centroids: {self.centroids}")
            # print(f"Cluster data: {[list(map(int, cluster)) for cluster in self.data]}")

    def predict(self, x):
        labels = []
        for datapoint in x:
            distances = [self._euclidean_distance(datapoint, self.centroids[j]) for j in range(self.num_clusters)]
            labels.append(np.argmin(distances))
        return np.array(labels)

# Generate synthetic 2D data
np.random.seed(10)
data = np.random.rand(1000, 2) * 50  # 20 data points in 2D space
print("Data:\n", data)

# Create KMeans instance and fit the model
kmeans = KMeans(num_clusters=3, threshold=0.1)
kmeans.fit(data, epochs=100)

# Predict labels
labels = kmeans.predict(data)
print("labels >>>", len(labels))

# Visualization
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', label='Data points')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], color='red', marker='X', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.colorbar()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
