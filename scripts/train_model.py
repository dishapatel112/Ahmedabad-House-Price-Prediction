import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb

warnings.filterwarnings("ignore")

print("\n Training Production-Grade Ahmedabad House Price Model..\n")

ROOT = Path(__file__).resolve().parent
DATA_PATH = (ROOT / ".." / "cleaned_ahmedabad.csv").resolve()
MODEL_OUT = (ROOT / ".." / "house_price_model_prod_v1.pkl").resolve()

# LOAD CLEANED DATA
df = pd.read_csv(DATA_PATH)



df["price_log"] = np.log1p(df["price_rupees"]) #price_log = log(price_rupees + 1)

furn_map = {"Unfurnished": 0.0, "Semi-Furnished": 0.6, "Furnished": 1.0}
df["furnishing_ord"] = df["furnishing"].map(furn_map)

# Ordinal Encode Property Status
enc = OrdinalEncoder()
df["status_enc"] = enc.fit_transform(df[["status"]])

# Target Encoding for Locality
locality_avg = df.groupby("locality")["price_per_sqft"].mean()
locality_map = locality_avg.to_dict()
global_avg = df["price_per_sqft"].mean()

df["locality_te"] = df["locality"].map(locality_map)
df["locality_te"].fillna(global_avg, inplace=True)

# Locality min–max ppsqft for realistic clipping
loc_min = df.groupby("locality")["price_per_sqft"].min().to_dict()
loc_max = df.groupby("locality")["price_per_sqft"].max().to_dict()

# Interaction Feature
df["loc_furn_inter"] = df["locality_te"] * df["furnishing_ord"]

# FEATURE SET
numeric_cols = ["bhk", "area_sqft", "locality_te", "furnishing_ord", "loc_furn_inter"]

X = df[numeric_cols + ["status_enc"]]
y = df["price_log"]

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# LIGHTGBM MODEL WITH MONOTONE
monotone_constraints = [1, 1, 1, 1, 1, 0]

model = lgb.LGBMRegressor(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2,
    reg_alpha=1.2,
    random_state=42,
    monotone_constraints=monotone_constraints,
    verbosity=-1       
)

model.fit(X_train, y_train)

# EVALUATION
preds = model.predict(X_test) #log price predict kre pela
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

preds_real = np.expm1(preds)
y_test_real = np.expm1(y_test)
rmse_rupees = np.sqrt(mean_squared_error(y_test_real, preds_real))

print(f" RMSE: {rmse:.4f}")
print(f" RMSE (₹ rupees): {rmse_rupees:,.0f}")
print(f" R² Score: {r2:.4f}")

# SAVE MODEL + ENCODERS
joblib.dump({
    "model": model,
    "encoder": enc,
    "numeric_cols": numeric_cols,
    "locality_map": locality_map,
    "locality_global": global_avg,
    "locality_min": loc_min,
    "locality_max": loc_max
}, MODEL_OUT)

print(f"\n Model saved as: {MODEL_OUT}\n")
