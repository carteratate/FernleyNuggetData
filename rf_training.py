import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

parser = argparse.ArgumentParser(description="Random forest training script")
parser.add_argument("--features", default="Data/features.csv", help="Input feature CSV")
args = parser.parse_args()

# === STEP 1: Load features ===
df = pd.read_csv(args.features)
y = df["coinin"]
X = df.drop(columns=["coinin"])

# === STEP 2: Track metrics and predictions ===
seeds = [0, 1, 2, 3, 4]
results = []

for seed in seeds:
    print(f"Running fold with seed {seed}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results.append({
        "seed": seed,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "y_true": y_test,
        "y_pred": y_pred
    })

# === STEP 3: Print average metrics ===
avg_rmse = np.mean([r["rmse"] for r in results])
avg_r2 = np.mean([r["r2"] for r in results])
avg_mae = np.mean([r["mae"] for r in results])

print("\n=== Average Performance Over 5 Splits ===")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average R²:   {avg_r2:.4f}")
print(f"Average MAE:  {avg_mae:.2f}")

# === STEP 4: Find fold with median R² ===
results_sorted = sorted(results, key=lambda x: x["r2"])
median_fold = results_sorted[len(results_sorted) // 2]

# === STEP 5: Plot Predicted vs Actual for median fold ===
plt.figure(figsize=(8, 6))
plt.scatter(median_fold["y_true"], median_fold["y_pred"], alpha=0.3)
plt.plot([0, max(median_fold["y_true"])], [0, max(median_fold["y_true"])], 'r--')
plt.xlabel("Actual Coin-In")
plt.ylabel("Predicted Coin-In")
plt.title(f"Random Forest: Predicted vs Actual (Seed {median_fold['seed']})\nR²: {median_fold['r2']:.4f} | MAE: {median_fold['mae']:.2f}")
# Trim x and y axes to focus on the main data range
plt.xlim([0, 2000])
plt.ylim([0, 2000])
plt.grid(True)
plt.tight_layout()
plt.show()



