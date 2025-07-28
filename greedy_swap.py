import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestRegressor
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from utils import assign_spatial_features

parser = argparse.ArgumentParser(description="Greedy layout optimizer")
parser.add_argument("--features", default="Data/features.csv", help="Training feature CSV")
parser.add_argument("--future-layout", default="Data/future_month_testdata.csv", help="Input future layout CSV")
parser.add_argument("--cluster-coords", default="Data/clustered_coordinates.csv", help="Cluster coordinates CSV")
parser.add_argument("--original-output", default="Data/original_layout_with_predictions.csv", help="Output CSV for initial predictions")
parser.add_argument("--swap-log", default="Data/swap_log.csv", help="CSV to record swap log")
parser.add_argument("--optimized-output", default="Data/optimized_layout_greedy.csv", help="Output CSV for optimized layout")
args = parser.parse_args()

# === Coin-in prediction ===
def predict_total_coinin(df, model):
    features = df[model.feature_names_in_]
    return model.predict(features)

# === Evaluate a candidate swap ===
def evaluate_swap(args):
    df, model, idx_a, idx_b, current_total = args
    df_temp = df.copy()
    df_temp.loc[[idx_a, idx_b], ["x", "y"]] = df_temp.loc[[idx_b, idx_a], ["x", "y"]].values
    df_temp = assign_spatial_features(df_temp)
    df_temp["coinin"] = predict_total_coinin(df_temp, model)
    new_total = df_temp["coinin"].sum()
    gain = new_total - current_total
    return gain, (idx_a, idx_b)

# === Greedy optimizer ===
def greedy_swap_optimizer(df, model, max_swaps=10, min_gain_threshold=25, log_every=1):
    current_df = df.copy()
    current_df["coinin"] = predict_total_coinin(current_df, model)
    current_total = current_df["coinin"].sum()

    swaps_done = 0
    swap_log = []
    swap_counts = {idx: 0 for idx in current_df.index}

    print(f"Initial predicted total coin-in: ${current_total:,.2f}")

    while swaps_done < max_swaps:
        if swaps_done % log_every == 0:
            print(f"\n--- Swap Round {swaps_done + 1} ---")

        # Recompute bar and non-bar machine sets after every swap
        bar_df = current_df[current_df["bar_slot"] == 1]
        nonbar_df = current_df[current_df["bar_slot"] == 0]

        candidate_args = []
        for idx_a in nonbar_df.index:
            for idx_b in nonbar_df.index:
                if idx_a >= idx_b:
                    continue
                candidate_args.append((current_df, model, idx_a, idx_b, current_total))

        for idx_a in bar_df.index:
            for idx_b in bar_df.index:
                if idx_a >= idx_b:
                    continue
                candidate_args.append((current_df, model, idx_a, idx_b, current_total))

        if swaps_done % log_every == 0:
            print(f"Evaluating {len(candidate_args):,} swap candidates...")

        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(executor.map(evaluate_swap, candidate_args))

        best_gain = 0
        best_pair = None
        for gain, pair in results:
            if gain > best_gain:
                best_gain = gain
                best_pair = pair

        if best_gain < min_gain_threshold:
            print(f"Stopping early: best possible gain ${best_gain:.2f} is below threshold.")
            break

        idx_a, idx_b = best_pair
        print(f"Swap {swaps_done + 1}: Gain = ${best_gain:.2f} from swapping index {idx_a} and {idx_b}")
        swap_log.append({"swap_num": swaps_done + 1, "index_a": idx_a, "index_b": idx_b, "gain": best_gain})
        swap_counts[idx_a] += 1
        swap_counts[idx_b] += 1

        current_df.loc[[idx_a, idx_b], ["x", "y"]] = current_df.loc[[idx_b, idx_a], ["x", "y"]].values
        current_df = assign_spatial_features(current_df)
        current_df["coinin"] = predict_total_coinin(current_df, model)
        current_total = current_df["coinin"].sum()
        swaps_done += 1

    return current_df, swap_log, swap_counts

# === MAIN ===
if __name__ == "__main__":
    train_df = pd.read_csv(args.features)
    y = train_df["coinin"]
    X = train_df.drop(columns=["coinin"])
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
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

    # Run optimization
    optimized_df, swap_log, swap_counts = greedy_swap_optimizer(
        future_df, model, max_swaps=10, min_gain_threshold=25, log_every=1
    )
    optimized_total = optimized_df["coinin"].sum()

    # === Save top swaps ===
    pd.DataFrame(swap_log).to_csv(args.swap_log, index=False)

    # === Heatmap: Swap frequency ===
    swap_freq = pd.Series(swap_counts).rename("count")
    swap_coords = optimized_df[["x", "y"]].copy()
    swap_coords["count"] = swap_coords.index.map(swap_freq).fillna(0)
    pivot_swap = swap_coords.pivot(index="x", columns="y", values="count")

    plt.figure(figsize=(12, 10))
    plt.imshow(pivot_swap, origin="upper", cmap="Reds", aspect="equal")
    plt.colorbar(label="Swap Frequency")
    plt.title("Swap Frequency Heatmap")
    plt.xlabel("Y coordinate")
    plt.ylabel("X coordinate")
    plt.tight_layout()
    plt.show()

    # === Heatmap: Coin-in before and after ===
    after_coinin = optimized_df[["x", "y", "coinin"]].copy().rename(columns={"coinin": "coinin_after"})
    comparison = original_coinin_per_machine.merge(after_coinin, on=["x", "y"], how="inner")

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

    # === Bar Plot: Before vs After ===
    plt.figure(figsize=(6, 4))
    plt.bar(["Original", "Optimized"], [original_total, optimized_total], color=["gray", "green"])
    plt.ylabel("Total Predicted Coin-In ($)")
    plt.title("Greedy Optimization Result")
    plt.tight_layout()
    plt.show()

    optimized_df.to_csv(args.optimized_output, index=False)
    print(f"\nSaved optimized layout to {args.optimized_output}")
    print(f"Original total predicted coin-in: ${original_total:,.2f}")
    print(f"Optimized total predicted coin-in: ${optimized_total:,.2f}")
