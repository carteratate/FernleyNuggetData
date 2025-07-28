import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def main() -> None:
    """Train several nonlinear models and compare them."""

    # === STEP 1: Load your features ===
    df = pd.read_csv("Data/features.csv")

    # === STEP 2: Define target and features ===
    y = df["coinin"]
    X = df.drop(columns=["coinin"])

    # === STEP 3-6: Loop over different seeds and collect performance ===
    seeds = [0, 1, 2, 3, 4]

    # For each model
    metrics = {
        "Random Forest": {"r2": [], "rmse": [], "y_true": [], "y_pred": []},
        "LightGBM": {"r2": [], "rmse": [], "y_true": [], "y_pred": []},
        "XGBoost": {"r2": [], "rmse": [], "y_true": [], "y_pred": []},
    }

    for seed in seeds:
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
        # === Random Forest ===
        # Best performing was standard, without tuning
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        # rf = RandomForestRegressor(
        #     n_estimators=300,
        #     min_samples_split=2,
        #     min_samples_leaf=1,
        #     max_features='sqrt',
        #     max_depth=40,
        #     random_state=42,
        #     n_jobs=-1
        # )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        metrics["Random Forest"]["r2"].append(r2_score(y_test, y_pred_rf))
        metrics["Random Forest"]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
        metrics["Random Forest"]["y_true"].extend(y_test)
        metrics["Random Forest"]["y_pred"].extend(y_pred_rf)
    
        # === LightGBM ===
        # lgb = LGBMRegressor(n_estimators=100, random_state=42)
        # Best performing LGBM parameters from tuning
        lgb = LGBMRegressor(
            subsample=0.8,
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            colsample_bytree=0.8,
            num_leaves=100,
            random_state=42,
            verbose=-1
        )
        lgb.fit(X_train, y_train)
        y_pred_lgb = lgb.predict(X_test)
        metrics["LightGBM"]["r2"].append(r2_score(y_test, y_pred_lgb))
        metrics["LightGBM"]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred_lgb)))
        metrics["LightGBM"]["y_true"].extend(y_test)
        metrics["LightGBM"]["y_pred"].extend(y_pred_lgb)
    
        # === XGBoost ===
        # xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        # Best performing XGBoost parameters from tuning
        xgb = XGBRegressor(
            subsample=0.8,
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            colsample_bytree=0.6,
            gamma=0.1,
            random_state=42,
            verbosity=0
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        metrics["XGBoost"]["r2"].append(r2_score(y_test, y_pred_xgb))
        metrics["XGBoost"]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
        metrics["XGBoost"]["y_true"].extend(y_test)
        metrics["XGBoost"]["y_pred"].extend(y_pred_xgb)
    
    # === STEP 7: Print average metrics ===
    print("\n=== Average Performance Over 5 Splits ===")
    for model in metrics:
        avg_rmse = np.mean(metrics[model]["rmse"])
        avg_r2 = np.mean(metrics[model]["r2"])
        print(f"\n[{model}]")
        print(f"Average RMSE: {avg_rmse:.2f}")
        print(f"Average R²: {avg_r2:.4f}")
    
    # === STEP 8: Plot predicted vs actual ===
    plt.figure(figsize=(18, 5))
    for i, model in enumerate(metrics.keys(), 1):
        plt.subplot(1, 3, i)
        plt.scatter(metrics[model]["y_true"], metrics[model]["y_pred"], alpha=0.4)
        min_val = min(metrics[model]["y_true"])
        max_val = max(metrics[model]["y_true"])
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.title(f"{model}: Predicted vs Actual")
        plt.xlabel("Actual Coin-In")
        plt.ylabel("Predicted Coin-In")
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
