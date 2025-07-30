# import argparse
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import random
# from sklearn.ensemble import RandomForestRegressor
# from concurrent.futures import ProcessPoolExecutor
# import multiprocessing
# from utils import assign_spatial_features

# parser = argparse.ArgumentParser(description="Greedy layout optimizer")
# parser.add_argument("--features", default="Data/features.csv", help="Training feature CSV")
# parser.add_argument("--future-layout", default="Data/future_month_testdata.csv", help="Input future layout CSV")
# parser.add_argument("--cluster-coords", default="Data/clustered_coordinates.csv", help="Cluster coordinates CSV")
# parser.add_argument("--original-output", default="Data/original_layout_with_predictions.csv", help="Output CSV for initial predictions")
# parser.add_argument("--swap-log", default="Data/swap_log.csv", help="CSV to record swap log")
# parser.add_argument("--optimized-output", default="Data/optimized_layout_greedy.csv", help="Output CSV for optimized layout")
# parser.add_argument("--optimized-output-wPred", default="Data/optimized_layout_with_predictions_greedy.csv", help="Output CSV for optimized layout with predictions")
# args = parser.parse_args()

# # === Coin-in prediction ===
# def predict_total_coinin(df, model):
#     features = df[model.feature_names_in_]
#     return model.predict(features)

# def evaluate_swap(args):
#     df, model, idx_a, idx_b, current_total = args
#     df_temp = df.copy()

#     # Swap only the x and y coordinates
#     df_temp.loc[[idx_a, idx_b], ["x", "y"]] = df_temp.loc[[idx_b, idx_a], ["x", "y"]].values

#     # Recalculate spatial features while preserving non-spatial features
#     spatial_features = ["bar_slot", "near_bar", "near_main_door", "near_back_door"] + \
#                        [f"is_cluster{cid}" for cid in range(13)]
#     df_temp = assign_spatial_features(df_temp)

#     # Ensure non-spatial features remain unchanged
#     non_spatial_features = [col for col in df.columns if col not in ["x", "y"] + spatial_features]
#     df_temp[non_spatial_features] = df[non_spatial_features]

#     # Predict coinin and calculate the gain
#     df_temp["coinin"] = predict_total_coinin(df_temp, model)
#     new_total = df_temp["coinin"].sum()
#     gain = new_total - current_total

#     return gain, (idx_a, idx_b)

# # === Greedy optimizer ===
# def greedy_swap_optimizer(df, model, max_swaps=10, min_gain_threshold=25, log_every=1):
#     """
#     Greedy swap optimizer that swaps locations for the entire month (all days) for each machine.
#     Only swaps bar machines with bar machines, and nonbar machines with nonbar machines.
#     """
#     current_df = df.copy()
#     current_df["coinin"] = predict_total_coinin(current_df, model)
#     current_total = current_df["coinin"].sum()

#     print(f"Initial predicted total coin-in: ${current_total:,.2f}")

#     swaps_done = 0
#     swap_log = []
#     swap_counts = {mid: 0 for mid in current_df["machine_id"].unique()}

#     # Get unique machines and their bar status
#     machine_bar_status = current_df.drop_duplicates("machine_id")[["machine_id", "bar_slot"]]
#     bar_machines = machine_bar_status[machine_bar_status["bar_slot"] == 1]["machine_id"].tolist()
#     nonbar_machines = machine_bar_status[machine_bar_status["bar_slot"] == 0]["machine_id"].tolist()

#     while swaps_done < max_swaps:
#         if swaps_done % log_every == 0:
#             print(f"\n--- Swap Round {swaps_done + 1} ---")

#         best_gain = 0
#         best_pair = None
#         temp_df = current_df.copy()

#         # Bar <-> Bar swaps (swap all days for each machine)
#         for i, m_id_a in enumerate(bar_machines):
#             for m_id_b in bar_machines[i+1:]:
#                 print(f"Evaluating swap between bar machines {m_id_a} and {m_id_b}...")
#                 mask_a = temp_df["machine_id"] == m_id_a
#                 mask_b = temp_df["machine_id"] == m_id_b
#                 # Save original locations
#                 x_a, y_a = temp_df.loc[mask_a, ["x", "y"]].values[0]
#                 x_b, y_b = temp_df.loc[mask_b, ["x", "y"]].values[0]
#                 # Swap locations for all days
#                 temp_df.loc[mask_a, ["x", "y"]] = [x_b, y_b]
#                 temp_df.loc[mask_b, ["x", "y"]] = [x_a, y_a]
#                 temp_df = assign_spatial_features(temp_df)
#                 temp_df["coinin"] = predict_total_coinin(temp_df, model)
#                 new_total = temp_df["coinin"].sum()
#                 gain = new_total - current_total
#                 if gain > best_gain:
#                     best_gain = gain
#                     best_pair = (m_id_a, m_id_b)
#                 # Undo swap
#                 temp_df.loc[mask_a, ["x", "y"]] = [x_a, y_a]
#                 temp_df.loc[mask_b, ["x", "y"]] = [x_b, y_b]

#         # Nonbar <-> Nonbar swaps (swap all days for each machine)
#         for i, m_id_a in enumerate(nonbar_machines):
#             for m_id_b in nonbar_machines[i+1:]:
#                 print(f"Evaluating swap between nonbar machines {m_id_a} and {m_id_b}...")
#                 mask_a = temp_df["machine_id"] == m_id_a
#                 mask_b = temp_df["machine_id"] == m_id_b
#                 x_a, y_a = temp_df.loc[mask_a, ["x", "y"]].values[0]
#                 x_b, y_b = temp_df.loc[mask_b, ["x", "y"]].values[0]
#                 temp_df.loc[mask_a, ["x", "y"]] = [x_b, y_b]
#                 temp_df.loc[mask_b, ["x", "y"]] = [x_a, y_a]
#                 temp_df = assign_spatial_features(temp_df)
#                 temp_df["coinin"] = predict_total_coinin(temp_df, model)
#                 new_total = temp_df["coinin"].sum()
#                 gain = new_total - current_total
#                 if gain > best_gain:
#                     best_gain = gain
#                     best_pair = (m_id_a, m_id_b)
#                 # Undo swap
#                 temp_df.loc[mask_a, ["x", "y"]] = [x_a, y_a]
#                 temp_df.loc[mask_b, ["x", "y"]] = [x_b, y_b]

#         # If no gain meets the threshold, stop early
#         if best_gain < min_gain_threshold or best_pair is None:
#             print(f"Stopping early: best possible gain ${best_gain:.2f} is below threshold.")
#             break

#         # Perform the best swap for all days
#         m_id_a, m_id_b = best_pair
#         mask_a = current_df["machine_id"] == m_id_a
#         mask_b = current_df["machine_id"] == m_id_b
#         x_a, y_a = current_df.loc[mask_a, ["x", "y"]].values[0]
#         x_b, y_b = current_df.loc[mask_b, ["x", "y"]].values[0]
#         current_df.loc[mask_a, ["x", "y"]] = [x_b, y_b]
#         current_df.loc[mask_b, ["x", "y"]] = [x_a, y_a]

#         # Recalculate spatial features and update coin-in
#         current_df = assign_spatial_features(current_df)
#         current_df["coinin"] = predict_total_coinin(current_df, model)
#         current_total = current_df["coinin"].sum()

#         # Update swap counts
#         swap_counts[m_id_a] += 1
#         swap_counts[m_id_b] += 1

#         # Log the swap
#         swaps_done += 1
#         swap_log.append({"swap_num": swaps_done, "machine_id_a": m_id_a, "machine_id_b": m_id_b, "gain": best_gain})

#         print(f"Swap {swaps_done}: Gain = ${best_gain:.2f} from swapping machine {m_id_a} and {m_id_b}")
#         print(f"Current total predicted coin-in: ${current_total:,.2f}")

#     print(f"\nOptimization complete. Total swaps: {swaps_done}")
#     print(f"Final predicted total coin-in: ${current_total:,.2f}")

#     return current_df, swap_log, swap_counts

# # === MAIN ===
# if __name__ == "__main__":
#     train_df = pd.read_csv(args.features)
#     y = train_df["coinin"]
#     X = train_df.drop(columns=["coinin"])
#     model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#     print("Training Random Forest Regressor on all data...")
#     model.fit(X, y)

#     future_df = pd.read_csv(args.future_layout)
#     clustered_coords = pd.read_csv(args.cluster_coords)
#     if "cluster_id" in future_df.columns:
#         future_df = future_df.drop(columns=["cluster_id"])
#     future_df = future_df.merge(clustered_coords, on=["x", "y"], how="left")
#     future_df = assign_spatial_features(future_df)
#     future_df["coinin"] = predict_total_coinin(future_df, model)
#     future_df.to_csv(args.original_output, index=False)
#     original_total = future_df["coinin"].sum()
#     original_coinin_per_machine = future_df[["x", "y", "coinin"]].copy()

#     # Run optimization
#     print("\nStarting greedy swap optimization...")
#     optimized_df, swap_log = greedy_swap_optimizer(
#         future_df, model, max_swaps=10, min_gain_threshold=25, log_every=1
#     )
#     optimized_total = optimized_df["coinin"].sum()

#     pd.DataFrame(swap_log).to_csv(args.swap_log, index=False)

#     # Heatmap: Coin-in before and after
#     after_coinin = optimized_df[["x", "y", "coinin"]].copy().rename(columns={"coinin": "coinin_after"})
#     comparison = original_coinin_per_machine.merge(after_coinin, on=["x", "y"], how="inner")

#     fig, ax = plt.subplots(1, 2, figsize=(14, 6))
#     pivot_before = comparison.pivot(index="x", columns="y", values="coinin")
#     pivot_after = comparison.pivot(index="x", columns="y", values="coinin_after")

#     im0 = ax[0].imshow(pivot_before, origin="upper", cmap="coolwarm", aspect="equal", vmin=0, vmax=np.nanpercentile(pivot_before.values, 85))
#     ax[0].set_title("Coin-In Before Optimization")
#     plt.colorbar(im0, ax=ax[0])

#     im1 = ax[1].imshow(pivot_after, origin="upper", cmap="coolwarm", aspect="equal", vmin=0, vmax=np.nanpercentile(pivot_after.values, 85))
#     ax[1].set_title("Coin-In After Optimization")
#     plt.colorbar(im1, ax=ax[1])

#     plt.tight_layout()
#     plt.show()

#     # Bar Plot: Before vs After
#     plt.figure(figsize=(6, 4))
#     plt.bar(["Original", "Optimized"], [original_total, optimized_total], color=["gray", "green"])
#     plt.ylabel("Total Predicted Coin-In ($)")
#     plt.title("Greedy Optimization Result")
#     plt.tight_layout()
#     plt.show()

#     # === Save optimized layout with predictions ===
#     optimized_df.to_csv(args.optimized_output_wPred, index=False)
#     print(f"\nSaved optimized layout to {args.optimized_output_wPred}")


#     # === STEP 6: Generate machine layout ===
#     # Dynamically determine theme columns based on their position in the DataFrame
#     theme_columns = optimized_df.iloc[:, 50:184].columns.tolist()  

#     # Extract x, y, and the theme where is_theme is true
#     machine_layout = optimized_df.loc[:, ["x", "y"] + theme_columns]

#     # Find the theme where is_theme is true for each machine
#     machine_layout["theme"] = machine_layout[theme_columns].idxmax(axis=1).str.replace("is_", "")

#     # Drop duplicates to ensure only one machine per x, y location
#     machine_layout = machine_layout.drop_duplicates(subset=["x", "y"])

#     # Keep only x, y, and theme columns
#     machine_layout = machine_layout.loc[:, ["x", "y", "theme"]]

#     # Save the machine layout to a separate CSV file
#     machine_layout.to_csv(args.optimized_output, index=False)



    
#     print(f"Original total predicted coin-in: ${original_total:,.2f}")
#     print(f"Optimized total predicted coin-in: ${optimized_total:,.2f}")


import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from utils import assign_spatial_features

parser = argparse.ArgumentParser(description="Greedy layout optimizer")
parser.add_argument("--features", default="Data/features.csv", help="Training feature CSV")
parser.add_argument("--future-layout", default="Data/future_month_testdata.csv", help="Input future layout CSV")
parser.add_argument("--cluster-coords", default="Data/clustered_coordinates.csv", help="Cluster coordinates CSV")
parser.add_argument("--original-output", default="Data/original_layout_with_predictions.csv", help="Output CSV for initial predictions")
parser.add_argument("--swap-log", default="Data/swap_log.csv", help="CSV to record swap log")
parser.add_argument("--optimized-output", default="Data/optimized_layout_greedy.csv", help="Output CSV for optimized layout")
parser.add_argument("--optimized-output-wPred", default="Data/optimized_layout_with_predictions_greedy.csv", help="Output CSV for optimized layout with predictions")
args = parser.parse_args()

def predict_total_coinin(df, model):
    features = df[model.feature_names_in_]
    return model.predict(features)

def greedy_swap_optimizer(df, model, max_swaps=10, min_gain_threshold=25, log_every=1):
    """
    Greedy swap optimizer that swaps locations for the entire month (all days) for each machine.
    Only swaps bar machines with bar machines, and nonbar machines with nonbar machines.
    Machines are identified by (theme, x, y) on the first day.
    """
    current_df = df.copy()
    current_df["coinin"] = predict_total_coinin(current_df, model)
    current_total = current_df["coinin"].sum()
    print(f"Initial predicted total coin-in: ${current_total:,.2f}")

    swaps_done = 0
    swap_log = []

    # Dynamically determine theme columns based on their position in the DataFrame
    theme_columns = current_df.iloc[:, 50:184].columns.tolist()

    # For each row, determine the theme
    current_df["theme"] = current_df[theme_columns].idxmax(axis=1).str.replace("is_", "")

    # Drop duplicates to get one row per unique machine (assumes x, y, theme uniquely identify a machine)
    unique_machines = current_df.drop_duplicates(subset=["x", "y", "theme"])[["theme", "x", "y", "bar_slot"]].copy()
    unique_machines["machine_key"] = list(zip(unique_machines["theme"], unique_machines["x"], unique_machines["y"]))

    bar_machines = unique_machines[unique_machines["bar_slot"] == 1]["machine_key"].tolist()
    nonbar_machines = unique_machines[unique_machines["bar_slot"] == 0]["machine_key"].tolist()

    current_df = current_df.drop(columns=["theme"])

    while swaps_done < max_swaps:
        if swaps_done % log_every == 0:
            print(f"\n--- Swap Round {swaps_done + 1} ---")

        best_gain = 0
        best_pair = None
        temp_df = current_df.copy()

        # Bar <-> Bar swaps
        for i, key_a in enumerate(bar_machines):
            print(f"Swap {swaps_done}: Evaluating swap between bar machines {key_a}...")
            for key_b in bar_machines[i+1:]:
                mask_a = (temp_df[theme_columns].idxmax(axis=1).str.replace("is_", "") == key_a[0]) & (temp_df["x"] == key_a[1]) & (temp_df["y"] == key_a[2])
                mask_b = (temp_df[theme_columns].idxmax(axis=1).str.replace("is_", "") == key_b[0]) & (temp_df["x"] == key_b[1]) & (temp_df["y"] == key_b[2])
                x_a, y_a = key_a[1], key_a[2]
                x_b, y_b = key_b[1], key_b[2]
                temp_df.loc[mask_a, ["x", "y"]] = [x_b, y_b]
                temp_df.loc[mask_b, ["x", "y"]] = [x_a, y_a]
                temp_df = assign_spatial_features(temp_df)
                temp_df["coinin"] = predict_total_coinin(temp_df, model)
                new_total = temp_df["coinin"].sum()
                gain = new_total - current_total
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (key_a, key_b)
                # Undo swap
                temp_df.loc[mask_a, ["x", "y"]] = [x_a, y_a]
                temp_df.loc[mask_b, ["x", "y"]] = [x_b, y_b]

        # Nonbar <-> Nonbar swaps
        for i, key_a in enumerate(nonbar_machines):
            print(f"Swap {swaps_done}: Evaluating swap between nonbar machines {key_a}...")
            for key_b in nonbar_machines[i+1:]:
                mask_a = (temp_df[theme_columns].idxmax(axis=1).str.replace("is_", "") == key_a[0]) & (temp_df["x"] == key_a[1]) & (temp_df["y"] == key_a[2])
                mask_b = (temp_df[theme_columns].idxmax(axis=1).str.replace("is_", "") == key_b[0]) & (temp_df["x"] == key_b[1]) & (temp_df["y"] == key_b[2])
                x_a, y_a = key_a[1], key_a[2]
                x_b, y_b = key_b[1], key_b[2]
                temp_df.loc[mask_a, ["x", "y"]] = [x_b, y_b]
                temp_df.loc[mask_b, ["x", "y"]] = [x_a, y_a]
                temp_df = assign_spatial_features(temp_df)
                temp_df["coinin"] = predict_total_coinin(temp_df, model)
                new_total = temp_df["coinin"].sum()
                gain = new_total - current_total
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (key_a, key_b)
                # Undo swap
                temp_df.loc[mask_a, ["x", "y"]] = [x_a, y_a]
                temp_df.loc[mask_b, ["x", "y"]] = [x_b, y_b]

        if best_gain < min_gain_threshold or best_pair is None:
            print(f"Stopping early: best possible gain ${best_gain:.2f} is below threshold.")
            break

        key_a, key_b = best_pair
        mask_a = (current_df[theme_columns].idxmax(axis=1).str.replace("is_", "") == key_a[0]) & (current_df["x"] == key_a[1]) & (current_df["y"] == key_a[2])
        mask_b = (current_df[theme_columns].idxmax(axis=1).str.replace("is_", "") == key_b[0]) & (current_df["x"] == key_b[1]) & (current_df["y"] == key_b[2])
        x_a, y_a = key_a[1], key_a[2]
        x_b, y_b = key_b[1], key_b[2]
        current_df.loc[mask_a, ["x", "y"]] = [x_b, y_b]
        current_df.loc[mask_b, ["x", "y"]] = [x_a, y_a]
        current_df = assign_spatial_features(current_df)
        current_df["coinin"] = predict_total_coinin(current_df, model)
        current_total = current_df["coinin"].sum()
        swaps_done += 1
        swap_log.append({
            "swap_num": swaps_done,
            "machine_a": key_a,
            "machine_b": key_b,
            "gain": best_gain
        })
        print(f"Swap {swaps_done}: Gain = ${best_gain:.2f} from swapping {key_a} and {key_b}")
        print(f"Current total predicted coin-in: ${current_total:,.2f}")

    print(f"\nOptimization complete. Total swaps: {swaps_done}")
    print(f"Final predicted total coin-in: ${current_total:,.2f}")

    return current_df, swap_log

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

    print("\nStarting greedy swap optimization...")
    optimized_df, swap_log = greedy_swap_optimizer(
        future_df, model, max_swaps=10, min_gain_threshold=25, log_every=1
    )
    optimized_total = optimized_df["coinin"].sum()

    pd.DataFrame(swap_log).to_csv(args.swap_log, index=False)

    after_coinin = optimized_df[["x", "y", "coinin"]].copy().rename(columns={"coinin": "coinin_after"})
    comparison = original_coinin_per_machine.merge(after_coinin, on=["x", "y"], how="inner")

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
    plt.title("Greedy Optimization Result")
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
