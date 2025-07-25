import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# === STEP 1: Load Coordinates Data ===
date_machine_path = "Data/Merged_By_Date_MachineID_Active.csv"

date_machine = pd.read_csv(date_machine_path)

# Extract coordinates (X, Y) from the DataFrame
coords = date_machine[['x', 'y']].values







# === STEP 2: Standardize (optional if X and Y are on same scale) ===
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)


# === STEP 3: Elbow Method to Find Best K ===
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(coords_scaled)
    inertias.append(kmeans.inertia_)

# Plot the elbow
plt.figure(figsize=(6, 4))
plt.plot(K_range, inertias, marker='o')
plt.title('Elbow Method to Determine Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (sum of squared distances)')
plt.grid(True)
plt.tight_layout()
plt.show()

# === STEP 4: Choose best K manually (or automate the elbow detection) ===
# For now, let's pick the "elbow" at k=5 (adjust based on plot)
best_k = 5

# === STEP 5: Final K-Means Clustering ===
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
clusters = final_kmeans.fit_predict(coords_scaled)

# === STEP 6: Plot Resulting Clusters ===
plt.figure(figsize=(7, 6))
for cluster_id in range(best_k):
    cluster_points = coords[clusters == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}', s=30)

# Optionally plot centroids
centroids = scaler.inverse_transform(final_kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='X', label='Centroids')

plt.title(f'K-Means Clustering with k={best_k}')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()