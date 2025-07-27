
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# === STEP 1: Load your features ===
df = pd.read_csv("Data/features.csv")

# === STEP 2: Define target and features ===
y = df["coinin"]
X = df.drop(columns=["coinin"])

# === STEP 3: Train/test split and model training ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# === STEP 4: Plot predicted vs actual ===
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
plt.xlabel("Actual Coin-In")
plt.ylabel("Predicted Coin-In")
plt.title("Random Forest: Predicted vs Actual (Single Split)")
plt.xlim([0, 2000])  # Zoom in
plt.ylim([0, 2000])
plt.grid(True)
plt.tight_layout()
plt.show()

# === STEP 5: Calculate and print RMSE R² ===
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
mae = mean_absolute_error(y_test, y_pred_rf)
print("Random Forest MAE:", mae)
print(f"Random Forest RMSE: {rmse_rf:.2f}")
print(f"Random Forest R²: {r2_rf:.4f}")


