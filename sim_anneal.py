import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from utils import assign_spatial_features
parser = argparse.ArgumentParser(description="Simulated annealing layout optimizer")
parser.add_argument("--features", default="Data/features.csv", help="Training feature CSV")
parser.add_argument("--future-layout", default="Data/future_month_layout.csv", help="Input future layout CSV")
parser.add_argument("--cluster-coords", default="Data/clustered_coordinates.csv", help="Cluster coordinates CSV")
parser.add_argument("--original-output", default="Data/original_layout_with_predictions.csv", help="Output CSV for initial predictions")
parser.add_argument("--optimized-output", default="Data/optimized_layout.csv", help="Output CSV for optimized layout")
args = parser.parse_args()

# === Helper to assign spatial features (based on updated positions) ===
# === Helper to predict coin-in ===
def predict_total_coinin(df, model):
    features = df[model.feature_names_in_]
    preds = model.predict(features)
    return preds

# === Simulated Annealing ===
def simulated_annealing(df, model, iterations=1000, swaps_per_iter=10, temp=1.0, cooling_rate=0.95):
    current_df = df.copy()
    current_df["coinin"] = predict_total_coinin(current_df, model)
    current_score = current_df["coinin"].sum()
    best_df = current_df.copy()
    best_score = current_score

    print(f"Initial predicted total coin-in: ${current_score:,.2f}")

    for i in range(iterations):
        temp *= cooling_rate
        temp_df = current_df.copy()

        # Identify bar and non-bar machines
        bar_mask = temp_df["bar_slot"] == 1
        bar_indices = temp_df[bar_mask].index.tolist()
        nonbar_indices = temp_df[~bar_mask].index.tolist()

        # Perform swaps
        for _ in range(swaps_per_iter):
            if random.random() < 0.5 and len(bar_indices) >= 2:
                a, b = random.sample(bar_indices, 2)
            else:
                a, b = random.sample(nonbar_indices, 2)

            temp_df.loc[a, ["x", "y"]], temp_df.loc[b, ["x", "y"]] = (
                temp_df.loc[b, ["x", "y"]].values,
                temp_df.loc[a, ["x", "y"]].values
            )

        # Update spatial features and reassign clusters
        temp_df = assign_spatial_features(temp_df)
        temp_df["coinin"] = predict_total_coinin(temp_df, model)
        new_score = temp_df["coinin"].sum()

        # Accept if better or by probability
        if new_score > current_score or random.random() < np.exp((new_score - current_score) / temp):
            current_df = temp_df
            current_score = new_score
            if new_score > best_score:
                best_df = temp_df
                best_score = new_score

        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}: Current Score = ${current_score:,.2f}, Best = ${best_score:,.2f}")

    return best_df, df["coinin"].sum(), best_score

# === MAIN EXECUTION ===

# Load training data and model
train_df = pd.read_csv(args.features)
y = train_df["coinin"]
X = train_df.drop(columns=["coinin"])
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# Load future layout
future_df = pd.read_csv(args.future_layout)

# Merge in cluster_id (ensure no suffixes)
clustered_coords = pd.read_csv(args.cluster_coords)
if "cluster_id" in future_df.columns:
    future_df = future_df.drop(columns=["cluster_id"])
future_df = future_df.merge(clustered_coords, on=["x", "y"], how="left")

# Assign all spatial features
future_df = assign_spatial_features(future_df)

# === Predict initial layout coin-in and save it ===
future_df = assign_spatial_features(future_df)
future_df["coinin"] = predict_total_coinin(future_df, model)

# Save original layout to CSV
future_df.to_csv(args.original_output, index=False)

# === Store original total for plotting ===
original_total = future_df["coinin"].sum()

future_df.to_csv(args.original_output, index=False)

# Run simulated annealing
optimized_df, original_total, optimized_total = simulated_annealing(
    future_df, model, iterations=1000, swaps_per_iter=10, temp=1.0, cooling_rate=0.95
)

# === Heatmap of Optimized Layout ===
agg = optimized_df.groupby(['x', 'y']).agg(total_wager=('coinin', 'sum')).reset_index()
pivot = agg.pivot(index='x', columns='y', values='total_wager')

plt.figure(figsize=(12, 10))
plt.imshow(
    pivot,
    origin='upper',
    cmap='coolwarm',
    aspect='equal',
    vmin=0,
    vmax=np.nanpercentile(pivot.values, 85)
)
plt.colorbar(label='Total Coin-In ($)')
plt.title("Optimized Casino Floor Heatmap (Predicted)")
plt.xlabel('Y coordinate (right)')
plt.ylabel('X coordinate (down)')
plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
plt.tight_layout()
plt.show()

# === Bar Plot: Original vs Optimized Coin-In ===
plt.figure(figsize=(6, 4))
plt.bar(["Original", "Optimized"], [original_total, optimized_total], color=["gray", "green"])
plt.ylabel("Total Predicted Coin-In ($)")
plt.title("Simulated Annealing Optimization Result")
plt.tight_layout()
plt.show()

# Save final layout
optimized_df.to_csv(args.optimized_output, index=False)
print(f"\nSaved optimized layout to {args.optimized_output}")
print(f"Original total predicted coin-in: ${original_total:,.2f}")
print(f"Optimized total predicted coin-in: ${optimized_total:,.2f}")

