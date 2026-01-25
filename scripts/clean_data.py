import pandas as pd
import numpy as np
import re
from pathlib import Path


RAW_CSV = "data/ahmedabad.csv"
OUT_CSV = "cleaned_ahmedabad.csv"

# ---------- Helper Functions ----------
def fix_encoding(s):
    if pd.isna(s):
        return ""
    s = str(s)
    return (
        s.replace("â‚¹", "₹")
         .replace("\xa0", " ")
         .replace("\u200e", "")
         .replace("\n", " ")
         .strip()
    )

def parse_price(s):
    if pd.isna(s):
        return np.nan
    s = fix_encoding(s).lower().replace(",", "").replace("rs.", "").replace("rs", "").replace("inr", "").strip()

    m = re.search(r"([\d\.]+)\s*(crore|cr)", s)
    if m:
        return float(m.group(1)) * 10000000

    m = re.search(r"([\d\.]+)\s*(lakh|lac|l)", s)
    if m:
        return float(m.group(1)) * 100000

    m = re.search(r"([\d\.]+)", s)
    if m:
        val = float(m.group(1))
        # User's logic preserved: assuming values < 1000 are in Lakhs by default
        return val * 100000 if val < 1000 else val

    return np.nan

def parse_area(s):
    if pd.isna(s):
        return np.nan
    s = fix_encoding(s).lower().replace(",", "")

    m = re.search(r'([\d\.]+)', s)
    if not m:
        return np.nan

    try:
        val = float(m.group(1))
    except:
        return np.nan

    if val == 0 and m.group(1) == ".":
        return np.nan

    if "sqyrd" in s or "sq yd" in s:
        return val * 9

    if "sqm" in s or "sq m" in s:
        return val * 10.7639

    return val

def extract_bhk(title):
    m = re.search(r"(\d+)\s*BHK", str(title))
    return int(m.group(1)) if m else np.nan

def extract_locality(title: str) -> str:
    
    if not isinstance(title, str):
        return ""

    title_clean = fix_encoding(title)
    candidate = ""

    # Look for phrase between 'in'/'at'/',' and 'Ahmedabad'/'Gujarat'
    # Yeh pehle word ya 2-3 words ko nikalega
    match = re.search(r"(?:in|at|,)?\s*([\w\s\-\.]+)[\s\.]+(?:Ahmedabad|A'bad|Gujarat)", title_clean, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().lower()
    else:
        #  Look for 'for Sale in [Locality]' pattern if Strategy 1 fails
        parts = title_clean.split(' for Sale in ')
        if len(parts) > 1:
            # First element after split, remove trailing commas/spaces
            candidate = parts[1].split(',')[0].strip().lower()
            
    if not candidate:
        return ""
    
    # Saare common project and generic terms ko hataya gaya hai
    bad_keywords = [
        "society", "residency", "apartment", "tower", "complex", "township",
        "arcade", "avenue", "status", "icon", "wave", "heights", "exotica",
        "flats", "flat", "metrocity", "bunglows", "appartment", "city", "ahmedabad",
        "gloria", "club", "house", "villa", "bungalow", "home", "prive", "pratham", 
        "palace", "signature", "luxury", "skylines", "terrace", "platinum",
        "cosmos", "elite", "garden", "resort", "project", "westlands", "vivanta", 
        "aura", "epitome", "amara", "kavisha", "infinity", "glow", "times",
        "residence", "corporate", "centra", "premium", "greens", "lakeview", 
        "park", "shanti", "plaza", "north", "south", "west", "east", "aurum",
        "shree", "pacifica", "skymark", "vivaan", "commercial", "homes"
    ]
    if any(word in candidate.split() for word in bad_keywords): # split() to match whole words
        return ""

    # Remove suffixes (like 'road', 'nagar', 'bypass') to merge similar areas 
    suffixes = [" road", " gam", " circle", " highway", " area", " locality", " nagar", " sec", " sector", " extension", " square", " bypass"] # Added "bypass"
    for suffix in suffixes:
        while candidate.endswith(suffix): 
             candidate = candidate.replace(suffix, "").strip()
    
    # Aggressive Merging and Normalization Rules (Compound Names) 
    merging_rules = {
        "ambli bopal": "Ambli",
        "bopal ambli": "Ambli",
        "iscon ambli": "Ambli",
        "isckon ambli": "Ambli",
        "ghuma bopal": "Bopal",
        "ghuma": "Bopal",
        "bopal ghuma": "Bopal",
        "south bopal": "Bopal", 
        "bopal south": "Bopal",
        "sola bhadaj": "Sola",
        "bhada": "Bhadaj", 
        "shilaj": "Shilaj",
        
        "nikol naroda": "Naroda",
        "nava naroda": "New Naroda",
        "new naroda": "New Naroda",
        "nava naroda road": "New Naroda",
      
        "nherynagar ambawadi": "Ambawadi",
        "nherynagar": "Ambawadi", # Treat as sub-locality
        "aambawadi": "Ambawadi", # Typo fix
        
        "bod": "Bodakdev", 
        "thal": "Thaltej", 
        "thalt": "Thaltej",
        "ranip": "Ranip", 
        "satellit": "Satellite", 
        "prahlad": "Prahlad Nagar", 
        "prahladnagar": "Prahlad Nagar",
        
        "sg highway": "SG Highway",
        "sarkhej gandhinagar": "SG Highway",
        "sarkhej": "Sarkhej",
        "gota": "Gota",
    }

    # Apply the merging and normalization rules
    normalized_candidate = candidate
    for key, normalized_name in merging_rules.items():
        # Using 'in' is important here to catch variations like "ambli bopal road"
        if key in normalized_candidate: 
            return normalized_name.title()

    # Final cleanup (replace multiple spaces with single space)
    candidate = re.sub(r'\s+', ' ', candidate).strip()

    # Filter out candidates that are too long (likely junk) or too short.
    if len(candidate.split()) > 3 or len(candidate) < 3:
         return ""
        
    # Remove any trailing non-alphabetic characters
    candidate = re.sub(r'[^a-zA-Z\s]+$', '', candidate).strip()
    
    # Filter out project names that survived the bad_keywords list
    if candidate.lower() in ["pacifica", "kavisha", "aurum", "shree", "skymark", "times", "vivaan", "atlantis", "sp"]:
         return ""
        
    return candidate.strip().title()

# Status Cleaner
def clean_status(x):
    if pd.isna(x):
        return "Under Construction"
    x = x.lower().strip()
    return "Ready To Move" if "ready" in x else "Under Construction"

# ---------- Main ----------
def main():
    try:
        df = pd.read_csv(RAW_CSV, dtype=str, encoding="utf-8", on_bad_lines="skip")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {RAW_CSV}.")
        return
        
    print("Loaded raw CSV:", df.shape)

    clean_rows = []

    for _, row in df.iterrows():
        title = fix_encoding(row.get("Title", ""))
        desc  = fix_encoding(row.get("description", ""))

        bhk = extract_bhk(title)
        area_sqft = parse_area(row.get("value_area", ""))
        price = parse_price(row.get("price", ""))

        locality = extract_locality(title)
        status = clean_status(fix_encoding(row.get("status", "")))

        full = (title + " " + desc + " " + fix_encoding(row.get("furnishing", ""))).lower()
        if "semi" in full:
            furnishing = "Semi-Furnished"
        elif "furnished" in full:
            furnishing = "Furnished"
        else:
            furnishing = "Unfurnished"

        if pd.isna(price) or pd.isna(area_sqft) or pd.isna(bhk):
            continue

        price_per_sqft = round(price / area_sqft, 2) if area_sqft > 0 else 0.0

        clean_rows.append({
            "bhk": int(bhk),
            "area_sqft": round(area_sqft, 2),
            "locality": locality,
            "furnishing": furnishing,
            "status": status,
            "price_rupees": round(price, 2),
            "price_per_sqft": price_per_sqft
        })

    df_clean = pd.DataFrame(clean_rows)

    # Drop empty or long locality names
    df_clean = df_clean[df_clean["locality"].str.split().str.len() <= 3].copy() 
    df_clean = df_clean[df_clean["locality"] != ""].copy()

    #  Drop duplicates
    df_clean = df_clean.drop_duplicates().copy()

    #  Minimum 4 entries wali localities  
    MIN_FREQUENCY = 4 
    locality_counts = df_clean['locality'].value_counts()
    frequent_localities = locality_counts[locality_counts >= MIN_FREQUENCY].index
    
    # Filter the DataFrame to keep only properties in frequent localities
    df_clean = df_clean[df_clean['locality'].isin(frequent_localities)].copy()
    print(f" Filtered out localities appearing less than {MIN_FREQUENCY} times. (More coverage)")

    # 3. Remove outliers in price_per_sqft
    q1 = df_clean["price_per_sqft"].quantile(0.01)
    q99 = df_clean["price_per_sqft"].quantile(0.99)
    df_clean = df_clean[(df_clean["price_per_sqft"] >= q1) & (df_clean["price_per_sqft"] <= q99)].copy()

    # Save cleaned data
    df_clean.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # Summary
    print("\n Summary (Strictly Cleaned and Normalized Locality Data - New Rules Applied):")
    print("Localities:", df_clean["locality"].nunique(), f"(Ab sirf >= {MIN_FREQUENCY} properties wali localities)")
    print("BHKs:", sorted(df_clean["bhk"].unique()))
    print("Price range:", df_clean["price_rupees"].min(), "to", df_clean["price_rupees"].max())
    print("Area range:", df_clean["area_sqft"].min(), "to", df_clean["area_sqft"].max())
    print("Cleaned dataset saved:", OUT_CSV)
    print("Final Shape:", df_clean.shape)
    
    print("\n Locality Distribution Top 15 - Saaf aur Accurate")
    print(df_clean['locality'].value_counts().head(15).to_string())
    
if __name__ == "__main__":
    main()