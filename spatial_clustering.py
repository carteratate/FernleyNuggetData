from ctdata_prep import create_sessions, create_machines, create_merged
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main() -> None:
    """Perform spatial clustering of machine coordinates."""
    
    # === STEP 1: Load Coordinates Data ===
    sessions = create_sessions()
    machines = create_machines()
    merged = create_merged(sessions, machines)
    
    # Extract coordinates (X, Y) from the DataFrame
    coord_merged = merged.drop_duplicates(subset=['x', 'y'])
    coords = coord_merged[['x', 'y']].values
    
    # === Identify bar machines ===
    is_at_bar = (coords[:, 0] > 30) & (coords[:, 1] < 20)  # x > 30 and y < 20
    bar_coords = coords[is_at_bar]
    other_coords = coords[~is_at_bar]
    
    # === Scale only the other_coords (not needed for raw visualization, but helps KMeans) ===
    scaler = StandardScaler()
    other_coords_scaled = scaler.fit_transform(other_coords)
    
    # === STEP 3: Elbow Method to Find Best K ===
    inertias = []
    K_range = range(2, 16)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(other_coords_scaled)
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
    
    # === Run K-Means on non-bar machines ===
    k_total = 13
    k_kmeans = k_total - 1
    
    kmeans = KMeans(n_clusters=k_kmeans, random_state=42, n_init='auto')
    other_labels = kmeans.fit_predict(other_coords_scaled)
    
    # === Combine bar machines (cluster 0) with others (shifted by +1) ===
    final_labels = np.empty(coords.shape[0], dtype=int)
    final_labels[is_at_bar] = 0                      # Cluster 0: bar machines
    final_labels[~is_at_bar] = other_labels + 1      # Clusters 1 to k_total-1
    
    # === Plot the results using modern colormap API ===
    base_cmap = plt.colormaps.get_cmap('tab20')  
    colors = base_cmap(np.linspace(0, 1, k_total))   
    
    plt.figure(figsize=(7, 6))
    for cluster_id in range(k_total):
        cluster_points = coords[final_labels == cluster_id]
        plt.scatter(cluster_points[:, 1], cluster_points[:, 0],
                    s=30, color=colors[cluster_id], label=f'Cluster {cluster_id}')
    
    plt.title('Manual + KMeans Clustering (Top-Left Origin)')
    plt.xlabel('Y Coordinate (→ Right)')
    plt.ylabel('X Coordinate (↓ Down)')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # === Add cluster labels to coord_merged DataFrame ===
    coord_merged = coord_merged.copy()
    coord_merged['cluster_id'] = final_labels
    
    # === Retain only necessary columns for coord_merged ===
    coord_merged = coord_merged[['x', 'y', 'cluster_id']]
    
    # === Add cluster_id to merged DataFrame by matching on (x, y) ===
    merged = merged.merge(coord_merged[['x', 'y', 'cluster_id']], on=['x', 'y'], how='left')
    
    # === Save both coord_merged and merged DataFrames to CSV ===
    coord_merged.to_csv("Data/clustered_coordinates.csv", index=False)
    merged.to_csv("Data/merged_with_clusters.csv", index=False)
    

if __name__ == "__main__":
    main()
