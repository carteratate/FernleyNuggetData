import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# === STEP 1: Parse arguments and load training data ===
parser = argparse.ArgumentParser(description="Predict coin-in for a future month")
parser.add_argument("--features", default="Data/features.csv", help="Path to training feature CSV")
parser.add_argument("--future-layout", default="Data/future_month_layout.csv", help="Path to future layout CSV")
args = parser.parse_args()

df_train = pd.read_csv(args.features)
y_train = df_train["coinin"]
X_train = df_train.drop(columns=["coinin"])

# === STEP 2: Train Random Forest on all data ===
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
print("Training Random Forest Regressor on all data...")
rf.fit(X_train, y_train)

# === STEP 3: Load future month layout ===
df_future = pd.read_csv(args.future_layout)
# Some future layouts may not already contain a ``coinin`` column.  In that
# case we simply use the DataFrame as-is when generating features.  Attempting
# to drop a missing column would raise a ``KeyError``.
if "coinin" in df_future.columns:
    X_future = df_future.drop(columns=["coinin"])
else:
    X_future = df_future.copy()
print("Predicting coin-in for future month layout...")
df_future["coinin"] = rf.predict(X_future)

# === STEP 4: Aggregate total coinin per (x, y) ===
agg = df_future.groupby(['x', 'y']).agg(total_wager=('coinin', 'sum')).reset_index()

# === STEP 5: Pivot and plot ===
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
plt.colorbar(label='Total Coin In ($)')
plt.title("Predicted Casino Floor Heatmap for Next Month")
plt.xlabel('Y coordinate (right)')
plt.ylabel('X coordinate (down)')
plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
plt.tight_layout()
plt.show()

# === STEP 6: Print total predicted coin-in for next month ===
total_monthly_coinin = df_future["coinin"].sum()
print(f"Total Predicted Coin-In for Next Month: ${total_monthly_coinin:,.2f}")

