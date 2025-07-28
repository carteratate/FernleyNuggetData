import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils import create_sessions, create_machines, create_merged

parser = argparse.ArgumentParser(description="Cluster slot machine coordinates")
parser.add_argument("--coord-output", default="Data/clustered_coordinates.csv", help="Output CSV for coordinates with clusters")
parser.add_argument("--merged-output", default="Data/merged_with_clusters.csv", help="Output CSV for merged data with clusters")
args = parser.parse_args()

session_map = {
    1: "asset",
    5: "vendor",
    6: "theme",
    12: "isprogressive",
    13: "holdpct",
    14: "denom",
    17: "x",
    18: "y",
    21: "time",
    24: "coinin",
    25: "coinout",
    28: "gamesplayed",
}

machine_map = {
    13: "serial",
    24: "multidenom",
    27: "maxbet",
    52: "x_coord",
    53: "y_coord",
}

sessions = create_sessions("Data/assetmeters 07-25-2025.csv", rename_map=session_map)
sessions = sessions[[
    "asset", "vendor", "theme", "isprogressive", "holdpct", "denom",
    "x", "y", "time", "coinin", "coinout", "gamesplayed",
]]
machines = create_machines("Data/TTgames00 07-25-2025.csv", rename_map=machine_map)
machines = machines[["serial", "multidenom", "maxbet", "x_coord", "y_coord"]]
merged = create_merged(sessions, machines, "asset", "serial")

coord_merged = merged.drop_duplicates(subset=['x', 'y'])
coords = coord_merged[['x', 'y']].values

is_at_bar = (coords[:, 0] > 30) & (coords[:, 1] < 20)
bar_coords = coords[is_at_bar]
other_coords = coords[~is_at_bar]

scaler = StandardScaler()
other_coords_scaled = scaler.fit_transform(other_coords)

inertias = []
K_range = range(2, 16)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(other_coords_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertias, marker='o')
plt.title('Elbow Method to Determine Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (sum of squared distances)')
plt.grid(True)
plt.tight_layout()
plt.show()

k_total = 13
k_kmeans = k_total - 1
kmeans = KMeans(n_clusters=k_kmeans, random_state=42, n_init='auto')
other_labels = kmeans.fit_predict(other_coords_scaled)

final_labels = np.empty(coords.shape[0], dtype=int)
final_labels[is_at_bar] = 0
final_labels[~is_at_bar] = other_labels + 1

base_cmap = plt.colormaps.get_cmap('tab20')
colors = base_cmap(np.linspace(0, 1, k_total))

plt.figure(figsize=(7, 6))
for cluster_id in range(k_total):
    cluster_points = coords[final_labels == cluster_id]
    plt.scatter(cluster_points[:, 1], cluster_points[:, 0], s=30, color=colors[cluster_id], label=f'Cluster {cluster_id}')

plt.title('Manual + KMeans Clustering (Top-Left Origin)')
plt.xlabel('Y Coordinate (→ Right)')
plt.ylabel('X Coordinate (↓ Down)')
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.show()

coord_merged = coord_merged.copy()
coord_merged['cluster_id'] = final_labels
coord_merged = coord_merged[['x', 'y', 'cluster_id']]
merged = merged.merge(coord_merged[['x', 'y', 'cluster_id']], on=['x', 'y'], how='left')

coord_merged.to_csv(args.coord_output, index=False)
merged.to_csv(args.merged_output, index=False)
