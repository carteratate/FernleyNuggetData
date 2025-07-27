import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# === STEP 1: Load your features ===
df = pd.read_csv("Data/features.csv")

# === STEP 2: Define target and features ===
y = df["coinin"]
X = df.drop(columns=["coinin"])

# === STEP 3: Split into train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 4: Random Forest Regressor ===
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\n[Random Forest Regressor]")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R²: {r2_rf:.4f}")

# === STEP 5: LightGBM Regressor ===
lgb = LGBMRegressor(n_estimators=100, random_state=42)
lgb.fit(X_train, y_train)
y_pred_lgb = lgb.predict(X_test)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
r2_lgb = r2_score(y_test, y_pred_lgb)

print("\n[LightGBM Regressor]")
print(f"RMSE: {rmse_lgb:.2f}")
print(f"R²: {r2_lgb:.4f}")

# === STEP 6: XGBoost Regressor ===
xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print("\n[XGBoost Regressor]")
print(f"RMSE: {rmse_xgb:.2f}")
print(f"R²: {r2_xgb:.4f}")


import matplotlib.pyplot as plt

# === STEP 7: Plot predicted vs actual for each model ===
plt.figure(figsize=(18, 5))

# Random Forest
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.title("Random Forest: Predicted vs Actual")
plt.xlabel("Actual Coin-In")
plt.ylabel("Predicted Coin-In")

# LightGBM
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_lgb, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.title("LightGBM: Predicted vs Actual")
plt.xlabel("Actual Coin-In")
plt.ylabel("Predicted Coin-In")

# XGBoost
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_xgb, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.title("XGBoost: Predicted vs Actual")
plt.xlabel("Actual Coin-In")
plt.ylabel("Predicted Coin-In")

plt.tight_layout()
plt.show()

# === STEP 8: Compare performance across models ===
models = ['Random Forest', 'LightGBM', 'XGBoost']
rmse_scores = [rmse_rf, rmse_lgb, rmse_xgb]
r2_scores = [r2_rf, r2_lgb, r2_xgb]

# Bar chart for RMSE
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(models, rmse_scores, color='skyblue')
plt.title("RMSE Comparison")
plt.ylabel("RMSE")

# Bar chart for R²
plt.subplot(1, 2, 2)
plt.bar(models, r2_scores, color='salmon')
plt.title("R² Comparison")
plt.ylabel("R²")

plt.tight_layout()
plt.show()
