# import pandas as pd
# import numpy as np
# import time
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# import lightgbm as lgb
# import xgboost as xgb

# # === STEP 1: Load your features ===
# df = pd.read_csv("Data/features.csv")

# # === STEP 2: Define target and features ===
# y = df["coinin"]
# X = df.drop(columns=["coinin"])

# # === STEP 3: Split into train/test ===
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # === RANDOM FOREST TUNING ===
# print("\n=== Random Forest Regressor (Hyperparameter Tuning) ===")
# start = time.time()
# rf = RandomForestRegressor(random_state=42)
# rf_param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt']
# }
# rf_random = RandomizedSearchCV(
#     estimator=rf,
#     param_distributions=rf_param_grid,
#     n_iter=25,
#     cv=5,
#     verbose=2,
#     random_state=42,
#     n_jobs=-1
# )
# rf_random.fit(X_train, y_train)
# end = time.time()
# y_pred_rf = rf_random.predict(X_test)
# print(f"\n[Random Forest] RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
# print(f"[Random Forest] R²: {r2_score(y_test, y_pred_rf):.4f}")
# print(f"[Random Forest] Training Time: {(end - start)/60:.2f} minutes")

# # === LIGHTGBM TUNING ===
# print("\n=== LightGBM Regressor (Hyperparameter Tuning) ===")
# start = time.time()
# lgb_model = lgb.LGBMRegressor(random_state=42)
# lgb_param_grid = {
#     'num_leaves': [31, 50, 100],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [100, 200, 500],
#     'max_depth': [-1, 10, 20],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0]
# }
# lgb_random = RandomizedSearchCV(
#     estimator=lgb_model,
#     param_distributions=lgb_param_grid,
#     n_iter=25,
#     cv=5,
#     verbose=2,
#     random_state=42,
#     n_jobs=-1
# )
# lgb_random.fit(X_train, y_train)
# end = time.time()
# y_pred_lgb = lgb_random.predict(X_test)
# print(f"\n[LightGBM] RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lgb)):.2f}")
# print(f"[LightGBM] R²: {r2_score(y_test, y_pred_lgb):.4f}")
# print(f"[LightGBM] Training Time: {(end - start)/60:.2f} minutes")

# # === XGBOOST TUNING ===
# print("\n=== XGBoost Regressor (Hyperparameter Tuning) ===")
# start = time.time()
# xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0)
# xgb_param_grid = {
#     'n_estimators': [100, 200, 500],
#     'max_depth': [3, 6, 10],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'gamma': [0, 0.1, 0.2]
# }
# xgb_random = RandomizedSearchCV(
#     estimator=xgb_model,
#     param_distributions=xgb_param_grid,
#     n_iter=25,
#     cv=5,
#     verbose=2,
#     random_state=42,
#     n_jobs=-1
# )
# xgb_random.fit(X_train, y_train)
# end = time.time()
# y_pred_xgb = xgb_random.predict(X_test)
# print(f"\n[XGBoost] RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.2f}")
# print(f"[XGBoost] R²: {r2_score(y_test, y_pred_xgb):.4f}")
# print(f"[XGBoost] Training Time: {(end - start)/60:.2f} minutes")

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

# === STEP 1: Load your features ===
df = pd.read_csv("Data/features.csv")

# === STEP 2: Define target and features ===
y = df["coinin"]
X = df.drop(columns=["coinin"])

# === STEP 3: Split into train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Baseline Comparison Function ===
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\n[{name}]")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    plt.hist(y_test - y_pred, bins=50)
    plt.title(f"Residuals for {name}")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.show()

# === RANDOM FOREST ===
print("\n=== Random Forest Regressor ===")
# Baseline
baseline_rf = RandomForestRegressor(n_estimators=100, random_state=42)
baseline_rf.fit(X_train, y_train)
evaluate_model("Random Forest Baseline", baseline_rf, X_test, y_test)

# Tuning
start = time.time()
rf = RandomForestRegressor(random_state=42)
rf_param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
rf_random = RandomizedSearchCV(rf, rf_param_grid, n_iter=50, cv=5, verbose=2,
                               random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
end = time.time()
print("Best Params:", rf_random.best_params_)
print("Best CV R²:", rf_random.best_score_)
evaluate_model("Random Forest Tuned", rf_random.best_estimator_, X_test, y_test)
print(f"Training Time: {(end - start)/60:.2f} minutes")

# === LIGHTGBM ===
print("\n=== LightGBM Regressor ===")
# Baseline
baseline_lgb = lgb.LGBMRegressor(n_estimators=100, random_state=42)
baseline_lgb.fit(X_train, y_train)
evaluate_model("LightGBM Baseline", baseline_lgb, X_test, y_test)

# Tuning
start = time.time()
lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_param_grid = {
    'num_leaves': [31, 50, 100],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'max_depth': [-1, 10, 20],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
lgb_random = RandomizedSearchCV(lgb_model, lgb_param_grid, n_iter=50, cv=5,
                                verbose=2, random_state=42, n_jobs=-1)
lgb_random.fit(X_train, y_train)
end = time.time()
print("Best Params:", lgb_random.best_params_)
print("Best CV R²:", lgb_random.best_score_)
evaluate_model("LightGBM Tuned", lgb_random.best_estimator_, X_test, y_test)
print(f"Training Time: {(end - start)/60:.2f} minutes")

# === XGBOOST ===
print("\n=== XGBoost Regressor ===")
# Baseline
baseline_xgb = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
baseline_xgb.fit(X_train, y_train)
evaluate_model("XGBoost Baseline", baseline_xgb, X_test, y_test)

# Tuning
start = time.time()
xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0)
xgb_param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}
xgb_random = RandomizedSearchCV(xgb_model, xgb_param_grid, n_iter=50, cv=5, 
                                verbose=2, random_state=42, n_jobs=-1)
def main() -> None:
    """Example hyperparameter tuning routine."""

    start = time.time()
    xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0)
    xgb_param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.2],
    }
    xgb_random = RandomizedSearchCV(
        xgb_model, xgb_param_grid, n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1
    )
    xgb_random.fit(X_train, y_train)
    end = time.time()
    print("Best Params:", xgb_random.best_params_)
    print("Best CV R²:", xgb_random.best_score_)
    evaluate_model("XGBoost Tuned", xgb_random.best_estimator_, X_test, y_test)
    print(f"Training Time: {(end - start)/60:.2f} minutes")


if __name__ == "__main__":
    main()

