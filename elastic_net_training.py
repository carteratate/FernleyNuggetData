import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor


def main() -> None:
    """Train several regression models and display their performance."""

    # === STEP 1: Load your features ===
    df = pd.read_csv("Data/features.csv")

    # === STEP 2: Define target and features ===
    # You are predicting coinin
    y = df["coinin"]

    # Drop the target and any other columns you don't want as features
    X = df.drop(columns=["coinin"])  # drop other target-like columns if needed (e.g., holdpct)

    # === STEP 3: Split into train/test ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # === STEP 4: Standardize features (ElasticNet benefits from this) ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === STEP 5: Train Elastic Net with cross-validation ===
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],  # Mix of L1/L2 ratios
        alphas=100,
        cv=5,
        max_iter=10000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # === STEP 6: Evaluate ===
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nBest Alpha: {model.alpha_}")
    print(f"Best L1 Ratio: {model.l1_ratio_}")
    print(f"RMSE on Test: {rmse:.2f}")
    print(f"R² on Test: {r2:.4f}")

    # === STEP 7: Review non-zero coefficients (optional) ===
    coef_df = pd.Series(model.coef_, index=X.columns)
    non_zero_coefs = coef_df[coef_df != 0].sort_values(key=abs, ascending=False)

    print("\nTop Predictive Features (Non-zero coefficients):\n")
    print(non_zero_coefs.head(20))

    # === STEP 8: Train RidgeCV for comparison ===
    alphas = np.logspace(-2, 2, 50)
    ridge = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X_train_scaled, y_train)

    y_pred_ridge = ridge.predict(X_test_scaled)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    r2_ridge = r2_score(y_test, y_pred_ridge)

    print("\n[RidgeCV]")
    print(f"Best Alpha: {ridge.alpha_}")
    print(f"RMSE: {rmse_ridge:.2f}")
    print(f"R²: {r2_ridge:.4f}")

    # === STEP 9: Train Random Forest for comparison ===
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    print("\n[Random Forest Regressor]")
    print(f"RMSE: {rmse_rf:.2f}")
    print(f"R²: {r2_rf:.4f}")


if __name__ == "__main__":
    main()


