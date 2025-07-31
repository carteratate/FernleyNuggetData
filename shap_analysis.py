import argparse
import os
import re

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="SHAP analysis for RandomForest")
parser.add_argument("--features", default="Data/features.csv", help="Features CSV")
parser.add_argument(
    "--output-dir",
    default="Images",
    help="Directory where summary plots will be written",
)
parser.add_argument(
    "--show",
    action="store_true",
    help="Display plots interactively in addition to saving",
)
parser.add_argument(
    "--shap-sample-size",
    type=int,
    default=1000,
    help="Number of rows to sample for SHAP analysis (default: 1000)",
)
args = parser.parse_args()

# === STEP 1: Load training features ===
# Same loading approach as rf_training.py lines 8-11
# which read the engineered feature set written by feat_create.py
# at lines 146-157.
df = pd.read_csv(args.features)
y = df["coinin"]
X = df.drop(columns=["coinin"])

# === STEP 2: Fit the RandomForest model ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training Random Forest Regressor on {len(X_train)} rows...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# === STEP 3: Compute SHAP values ===
# print("Computing SHAP values...")
# explainer = shap.TreeExplainer(rf)
# shap_values = explainer.shap_values(X_train)
print("Sampling data for SHAP analysis...")
if args.shap_sample_size < len(X_train):
    shap_X = X_train.sample(n=args.shap_sample_size, random_state=42)
else:
    shap_X = X_train
print(f"Computing SHAP values for {len(shap_X)} rows...")
explainer = shap.TreeExplainer(rf)
print("Explainer initialized. Computing SHAP values...")
shap_values = explainer.shap_values(shap_X)
print("Finished SHAP value computation.")

# === STEP 4: Summary plot for all features ===
print("Generating overall feature importance summary plot...")
os.makedirs(args.output_dir, exist_ok=True)
shap.summary_plot(shap_values, shap_X, show=False)
plt.title("Overall Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "shap_overall.png"))
if args.show:
    plt.show()
plt.close()


def categorize_columns(cols):
    """Group column names into spatial, game, and temporal categories."""

    categories = {"spatial": [], "game": [], "temporal": []}

    spatial_patterns = re.compile(
        r"(^x$|^y$|_coord$|near_|door|bar|^is_cluster|cluster|bar_slot)",
        re.IGNORECASE,
    )
    temporal_patterns = re.compile(
        r"(date|day|month|year|week|^is_(?:monday|tuesday|wednesday|thursday|friday|"
        r"saturday|sunday|weekend|january|february|march|april|may|june|july|august|"
        r"september|october|november|december|20\d{2}))",
        re.IGNORECASE,
    )
    game_patterns = re.compile(
        r"(game|theme|denom|bet|hold|payback|asset|vendor|multidenom|multigame|"
        r"maxbet|strength)",
        re.IGNORECASE,
    )

    for c in cols:
        lc = c.lower()
        if spatial_patterns.search(lc):
            categories["spatial"].append(c)
        elif temporal_patterns.search(lc):
            categories["temporal"].append(c)
        elif game_patterns.search(lc):
            categories["game"].append(c)

    return categories

print("Categorizing features into spatial, game, and temporal categories...")
categories = categorize_columns(shap_X.columns)

print("Feature categories identified:")
for name, cols in categories.items():
    if not cols:
        print(f"No {name} features found")
        continue
    print(f"\n{name.capitalize()} features found:", cols)
    shap.summary_plot(
        shap_values[:, [shap_X.columns.get_loc(c) for c in cols]],
        shap_X[cols],
        show=False,
    )
    plt.title(f"{name.capitalize()} Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"shap_{name}.png"))

    if args.show:
        plt.show()
    plt.close()
