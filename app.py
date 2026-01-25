from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "house_price_model_prod_v1.pkl"
DATA_PATH = ROOT / "cleaned_ahmedabad.csv"

# LOAD MODEL + ENCODERS
saved = joblib.load(MODEL_PATH)
model = saved["model"]
encoder = saved["encoder"]
numeric_cols = saved["numeric_cols"]
locality_map = saved["locality_map"]
locality_global = saved["locality_global"]
locality_min = saved["locality_min"]
locality_max = saved["locality_max"]

# LOAD DATA FOR DROPDOWNS
df = pd.read_csv(DATA_PATH)

localities = sorted(df["locality"].unique())
furnishings = ["Unfurnished", "Semi-Furnished", "Furnished"]
statuses = sorted(df["status"].unique())

# Furnishing numerical mapping
furn_map = {"Unfurnished": 0.0, "Semi-Furnished": 0.6, "Furnished": 1.0}


# PRE-CALCULATE BHK-LOCALITY PRICE RANGE 
PRICE_COLUMN = 'price_rupees' 

# 'price_rupees' min/max price 
bhk_locality_ranges = df.groupby(["bhk", "locality"])[PRICE_COLUMN].agg(["min", "max"]).to_dict("index")

# Global Fallback Range 
GLOBAL_MIN_PRICE = df[PRICE_COLUMN].min()
GLOBAL_MAX_PRICE = df[PRICE_COLUMN].max()

def format_inr(n): # for ₹  stuctre ke liye 
    n = int(n)
    s = str(n)
    if len(s) <= 3:
        return f"₹ {s}"
    last3 = s[-3:]
    s = s[:-3]
    parts = []
    while len(s) > 2:
        parts.append(s[-2:])
        s = s[:-2]
    if s:
        parts.append(s)
    return "₹ " + ",".join(parts[::-1]) + "," + last3


# EXTRA PRICE ADJUSTMENT 
def apply_furnishing_delta(price, furnishing):
    if furnishing == "Semi-Furnished":
        return price * 1.02
    elif furnishing == "Furnished":
        return price * 1.05
    return price



# NORMAL HOME ROUTE 
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    prediction_range = None
    error_msg = None
    price = None 
    min_price = None 
    max_price = None 
    bhk = None 
    locality = None 

    if request.method == "POST":
        try:
            bhk = float(request.form["bhk"])
            area_sqft = float(request.form["area_sqft"])
            locality = request.form["locality"]
            status = request.form["status"]
            furnishing_input = request.form["furnishing"]

            if bhk < 1 or area_sqft < 250:
                raise ValueError("BHK or Area is too small.")
            if bhk > 9 or area_sqft > 12000:
                raise ValueError("BHK or Area value is unrealistically high.")
            
            
            #  CALCULATION BASED ON BHK + LOCALITY
            lookup_key = (bhk, locality)
            price_range_data = bhk_locality_ranges.get(lookup_key)

            if price_range_data:
                min_price = price_range_data["min"]
                max_price = price_range_data["max"]
            else:
                min_price = GLOBAL_MIN_PRICE 
                max_price = GLOBAL_MAX_PRICE 
                
            # MODEL PREDICTION
            locality_te = locality_map.get(locality, locality_global)
            status_enc = encoder.transform([[status]])[0][0]
            furn_val = furn_map[furnishing_input]
            inter = locality_te * furn_val

            X = pd.DataFrame([{
                "bhk": bhk,
                "area_sqft": area_sqft,
                "locality_te": locality_te,
                "furnishing_ord": furn_val,
                "loc_furn_inter": inter,
                "status_enc": status_enc
            }])[numeric_cols + ["status_enc"]]

            pred_log = model.predict(X)[0] #LightGBM model log-price predict kare 
            price = np.expm1(pred_log)#Log price ne original rupees ma convert kare predicted_price_rupees = exp(predicted_log_price) - 1

            price = np.clip(price, min_price, max_price) 
            price = apply_furnishing_delta(price, furnishing_input)

            prediction = format_inr(price)
            
            prediction_range = (
                f"Market Range for {int(bhk)} BHK in {locality}: "
                f"{format_inr(min_price)} – {format_inr(max_price)}"
            )

        except Exception as e:
            error_msg = str(e)

    return render_template(
        "index.html",
        localities=localities,
        furnishings=furnishings,
        statuses=statuses,
        prediction=prediction,
        prediction_raw=price if prediction else None,
        min_price_raw=min_price if prediction else None,
        max_price_raw=max_price if prediction else None,
        prediction_range=prediction_range,
        error_msg=error_msg
)



# JSON ROUTE FOR JAVASCRIPT + CHART
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()

        bhk = float(data["bhk"])
        area_sqft = float(data["area_sqft"])
        locality = data["locality"]
        status = data["status"]
        furnishing_input = data["furnishing"]

        locality_te = locality_map.get(locality, locality_global)
        status_enc = encoder.transform([[status]])[0][0]
        furn_val = furn_map[furnishing_input]
        inter = locality_te * furn_val

        pps_min = locality_min.get(locality, 1500)
        pps_max = locality_max.get(locality, 20000)

        min_price = area_sqft * pps_min
        max_price = area_sqft * pps_max

        X = pd.DataFrame([{
            "bhk": bhk,
            "area_sqft": area_sqft,
            "locality_te": locality_te,
            "furnishing_ord": furn_val,
            "loc_furn_inter": inter,
            "status_enc": status_enc
        }])[numeric_cols + ["status_enc"]]

        pred_log = model.predict(X)[0]
        price = np.expm1(pred_log)

        price = np.clip(price, min_price, max_price)
        price = apply_furnishing_delta(price, furnishing_input)

        return jsonify({
            "predicted_price": int(price)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# RUN SERVER
if __name__ == "__main__":
    app.run(debug=True)