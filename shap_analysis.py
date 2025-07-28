import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# === STEP 1: Load training features ===
# Same loading approach as rf_training.py lines 8-11
# which read the engineered feature set written by feat_create.py
# at lines 146-157.
df = pd.read_csv("Data/features.csv")
y = df["coinin"]
X = df.drop(columns=["coinin"])

# === STEP 2: Fit the RandomForest model ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# === STEP 3: Compute SHAP values ===
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)

# === STEP 4: Summary plot ===
shap.summary_plot(shap_values, X_train)

# === STEP 5: Inspect spatial features ===
spatial_cols = [
    c for c in X_train.columns
    if c in [
        "x",
        "y",
        "near_main_door",
        "near_back_door",
        "near_bar",
        "bar_slot",
    ] or c.startswith("is_cluster")
]

if spatial_cols:
    print("\nSpatial features found:", spatial_cols)
    shap.summary_plot(
        shap_values[:, [X_train.columns.get_loc(c) for c in spatial_cols]],
        X_train[spatial_cols],
        plot_type="bar",
        show=False,
    )
    plt.title("Spatial Feature Importance")
    plt.show()
else:
    print("No spatial features present in the dataset.")
