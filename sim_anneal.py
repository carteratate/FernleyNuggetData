import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestRegressor
from utils import assign_spatial_features

parser = argparse.ArgumentParser(description="Simulated annealing layout optimizer")
parser.add_argument("--features", default="Data/features.csv", help="Training feature CSV")
parser.add_argument("--future-layout", default="Data/future_month_testdata.csv", help="Input future layout CSV")
parser.add_argument("--cluster-coords", default="Data/clustered_coordinates.csv", help="Cluster coordinates CSV")
parser.add_argument("--original-output", default="Data/original_layout_with_predictions.csv", help="Output CSV for initial predictions")
parser.add_argument("--optimized-output", default="Data/optimized_layout_simanneal.csv", help="Output CSV for optimized layout")
parser.add_argument("--optimized-output-wPred", default="Data/optimized_layout_with_predictions_simanneal.csv", help="Output CSV for optimized layout with predictions")
args = parser.parse_args()

def predict_total_coinin(df, model):
    features = df[model.feature_names_in_]
    return model.predict(features)

def simulated_annealing_swap_optimizer(df, model, iterations=1000, temp=1.0, cooling_rate=0.95, swaps_per_iter=1, log_every=50):
    """
    Simulated annealing optimizer that swaps locations for the entire month (all days) for each machine.
    Only swaps bar machines with bar machines, and nonbar machines with nonbar machines.
    Machines are identified by (theme, x, y).
    """
    current_df = df.copy()
    theme_columns = current_df.iloc[:, 50:184].columns.tolist()
    current_df["theme"] = current_df[theme_columns].idxmax(axis=1).str.replace("is_", "")

    # Drop duplicates to get one row per unique machine (assumes x, y, theme uniquely identify a machine)
    unique_machines = current_df.drop_duplicates(subset=["x", "y", "theme"])[["theme", "x", "y", "bar_slot"]].copy()
    unique_machines["machine_key"] = list(zip(unique_machines["theme"], unique_machines["x"], unique_machines["y"]))
    bar_machines = unique_machines[unique_machines["bar_slot"] == 1]["machine_key"].tolist()
    nonbar_machines = unique_machines[unique_machines["bar_slot"] == 0]["machine_key"].tolist()

    current_df = current_df.drop(columns=["theme"])
    current_df["coinin"] = predict_total_coinin(current_df, model)
    current_total = current_df["coinin"].sum()
    best_df = current_df.copy()
    best_total = current_total

    print(f"Initial predicted total coin-in: ${current_total:,.2f}")

    swap_log = []

    for it in range(iterations):
        print(f"Iteration {it + 1}/{iterations} | Current total: ${current_total:,.2f} | Temp: {temp:.4f}")
        temp *= cooling_rate
        improved = False
        temp_df = current_df.copy()

        for _ in range(swaps_per_iter):
            print(f"Performing {swaps_per_iter} swaps...")
            # Choose bar or nonbar group randomly
            if random.random() < 0.5 and len(bar_machines) >= 2:
                machine_list = bar_machines
                group = "bar"
            elif len(nonbar_machines) >= 2:
                machine_list = nonbar_machines
                group = "nonbar"
            else:
                continue

            key_a, key_b = random.sample(machine_list, 2)

            # Create masks for all rows for each machine (all days)
            mask_a = (temp_df[theme_columns].idxmax(axis=1).str.replace("is_", "") == key_a[0]) & (temp_df["x"] == key_a[1]) & (temp_df["y"] == key_a[2])
            mask_b = (temp_df[theme_columns].idxmax(axis=1).str.replace("is_", "") == key_b[0]) & (temp_df["x"] == key_b[1]) & (temp_df["y"] == key_b[2])
            x_a, y_a = key_a[1], key_a[2]
            x_b, y_b = key_b[1], key_b[2]

            # Swap locations for all days
            temp_df.loc[mask_a, ["x", "y"]] = [x_b, y_b]
            temp_df.loc[mask_b, ["x", "y"]] = [x_a, y_a]

        temp_df = assign_spatial_features(temp_df)
        temp_df["coinin"] = predict_total_coinin(temp_df, model)
        new_total = temp_df["coinin"].sum()
        delta = new_total - current_total

        # Accept swap if better, or with probability exp(delta/temp)
        if delta > 0 or random.random() < np.exp(delta / temp):
            current_df = temp_df
            current_total = new_total
            improved = True
            swap_log.append({
                "iteration": it + 1,
                "gain": delta,
                "accepted": True
            })
            if current_total > best_total:
                best_total = current_total
                best_df = current_df.copy()
        else:
            swap_log.append({
                "iteration": it + 1,
                "gain": delta,
                "accepted": False
            })

        if (it + 1) % log_every == 0 or it == 0:
            print(f"Iteration {it + 1}: Current total predicted coin-in: ${current_total:,.2f} | Best: ${best_total:,.2f} | Temp: {temp:.4f}")

    print(f"\nOptimization complete. Best predicted total coin-in: ${best_total:,.2f}")
    return best_df, swap_log

if __name__ == "__main__":
    train_df = pd.read_csv(args.features)
    y = train_df["coinin"]
    X = train_df.drop(columns=["coinin"])
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    print("Training Random Forest Regressor on all data...")
    model.fit(X, y)

    future_df = pd.read_csv(args.future_layout)
    clustered_coords = pd.read_csv(args.cluster_coords)
    if "cluster_id" in future_df.columns:
        future_df = future_df.drop(columns=["cluster_id"])
    future_df = future_df.merge(clustered_coords, on=["x", "y"], how="left")
    future_df = assign_spatial_features(future_df)
    future_df["coinin"] = predict_total_coinin(future_df, model)
    future_df.to_csv(args.original_output, index=False)
    original_total = future_df["coinin"].sum()
    original_coinin_per_machine = future_df[["x", "y", "coinin"]].copy()

    print("\nStarting simulated annealing optimization...")
    optimized_df, swap_log = simulated_annealing_swap_optimizer(
        future_df, model, iterations=500, temp=1.0, cooling_rate=0.98, swaps_per_iter=1, log_every=25
    )
    optimized_total = optimized_df["coinin"].sum()

    pd.DataFrame(swap_log).to_csv(args.optimized_output.replace(".csv", "_swap_log.csv"), index=False)

    # Aggregate before merging for heatmap
    original_coinin_per_machine = future_df.groupby(["x", "y"], as_index=False)["coinin"].sum()
    after_coinin = optimized_df.groupby(["x", "y"], as_index=False)["coinin"].sum().rename(columns={"coinin": "coinin_after"})
    comparison = original_coinin_per_machine.merge(after_coinin, on=["x", "y"], how="inner")

    # Draw heatmap of coin-in before and after optimization
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    pivot_before = comparison.pivot(index="x", columns="y", values="coinin")
    pivot_after = comparison.pivot(index="x", columns="y", values="coinin_after")

    im0 = ax[0].imshow(pivot_before, origin="upper", cmap="coolwarm", aspect="equal", vmin=0, vmax=np.nanpercentile(pivot_before.values, 85))
    ax[0].set_title("Coin-In Before Optimization")
    plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(pivot_after, origin="upper", cmap="coolwarm", aspect="equal", vmin=0, vmax=np.nanpercentile(pivot_after.values, 85))
    ax[1].set_title("Coin-In After Optimization")
    plt.colorbar(im1, ax=ax[1])

    plt.tight_layout()
    plt.show()

    # Bar Plot: Before vs After
    plt.figure(figsize=(6, 4))
    plt.bar(["Original", "Optimized"], [original_total, optimized_total], color=["gray", "green"])
    plt.ylabel("Total Predicted Coin-In ($)")
    plt.title("Simulated Annealing Optimization Result")
    plt.tight_layout()
    plt.show()

    optimized_df.to_csv(args.optimized_output_wPred, index=False)
    print(f"\nSaved optimized layout to {args.optimized_output_wPred}")

    # === STEP 6: Generate machine layout ===
    theme_columns = optimized_df.iloc[:, 50:184].columns.tolist()
    machine_layout = optimized_df.loc[:, ["x", "y"] + theme_columns]
    machine_layout["theme"] = machine_layout[theme_columns].idxmax(axis=1).str.replace("is_", "")
    machine_layout = machine_layout.drop_duplicates(subset=["x", "y"])
    machine_layout = machine_layout.loc[:, ["x", "y", "theme"]]
    machine_layout.to_csv(args.optimized_output, index=False)

    print(f"Original total predicted coin-in: ${original_total:,.2f}")
    print(f"Optimized total predicted coin-in: ${optimized_total:,.2f}")