import pandas as pd
import os
import time
import subprocess
import sys

# === Paths
PRED_CSV = "predictions/tft_forecast.csv"
SALES_CSV = "data/sales_train_validation.csv"
SALES_PARQUET = "data/sales_train_validation.parquet"
READY_CSV = "data/processed/m5_tft_ready.csv"
ID_MAP_CSV = "data/product_id_map.csv"
OUTPUT_ENRICHED = "predictions/tft_forecast_enriched.csv"

start = time.time()
print("📦 Starting metadata enrichment...")

# === Step 1: Ensure prediction exists
if not os.path.exists(PRED_CSV):
    raise FileNotFoundError(f"{PRED_CSV} not found. Run predict_tft.py first.")
pred_df = pd.read_csv(PRED_CSV)

# === Step 2: Load sales metadata
if os.path.exists(SALES_PARQUET):
    sales_meta = pd.read_parquet(SALES_PARQUET)
    print("⚡ Loaded sales metadata from Parquet")
else:
    print("📄 Converting sales metadata to Parquet...")
    sales_meta = pd.read_csv(SALES_CSV, usecols=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
    sales_meta.to_parquet(SALES_PARQUET)
    print("✅ Saved sales metadata as Parquet")

# === Step 3: Restore full product_id using original mapping
print("🔁 Decoding product_id using original training data...")
df_raw = pd.read_csv(READY_CSV)
df_raw["product_id"] = df_raw["product_id"].astype("category")
category_labels = df_raw["product_id"].cat.categories
product_id_map = pd.DataFrame({
    "product_id_code": list(range(len(category_labels))),
    "product_id_full": category_labels
})
alphabet_map = {chr(65 + i): i for i in range(len(category_labels))}

# Prepare enriched DataFrame
enriched_df = pred_df.copy()
enriched_df["product_id_code"] = enriched_df["product_id"].map(alphabet_map)
enriched_df = enriched_df.merge(product_id_map, on="product_id_code", how="left")

if enriched_df["product_id_full"].isnull().any():
    raise ValueError("❌ Could not map some product_id codes to full values.")

# === Step 4: Parse product_id_full
print("🔍 Parsing full product_id format...")
split_parts = enriched_df["product_id_full"].str.split("_", expand=True)
n_parts = split_parts.shape[1]

if n_parts == 5:
    split_parts.columns = ["state", "store_num", "cat_id", "cat_num", "item_suffix"]
    enriched_df["store_id"] = split_parts["state"] + "_" + split_parts["store_num"]
    enriched_df["cat_id"] = split_parts["cat_id"]
    enriched_df["item_id"] = split_parts["cat_id"] + "_" + split_parts["item_suffix"]
elif n_parts == 4:
    split_parts.columns = ["state", "store_num", "cat_id", "item_suffix"]
    enriched_df["store_id"] = split_parts["state"] + "_" + split_parts["store_num"]
    enriched_df["cat_id"] = split_parts["cat_id"]
    enriched_df["item_id"] = split_parts["cat_id"] + "_" + split_parts["item_suffix"]
elif n_parts == 3:
    split_parts.columns = ["state", "store_num", "cat_id"]
    enriched_df["store_id"] = split_parts["state"] + "_" + split_parts["store_num"]
    enriched_df["cat_id"] = split_parts["cat_id"]
    enriched_df["item_id"] = enriched_df["cat_id"] + "_GEN"
else:
    raise ValueError(f"Unsupported product_id format. Got {n_parts} parts. Sample: {enriched_df['product_id_full'].iloc[0]}")

# === Step 5: Build M5-compliant ID
enriched_df["id"] = enriched_df["item_id"] + "_" + enriched_df["store_id"] + "_validation"

# === Step 6: Merge with metadata
filtered_meta = sales_meta[sales_meta["id"].isin(enriched_df["id"].unique())]
final_df = enriched_df.merge(filtered_meta, on="id", how="left")

# === Step 7: Clean duplicate column names

# Remove exact duplicates (by column content)
final_df = final_df.loc[:, ~final_df.T.duplicated()]

# Deduplicate remaining column names
def deduplicate_columns(columns):
    seen = {}
    new_cols = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
    return new_cols

final_df.columns = deduplicate_columns(final_df.columns)

# === Step 8: Save final output
os.makedirs("predictions", exist_ok=True)
final_df.to_csv(OUTPUT_ENRICHED, index=False)
print(f"✅ Enriched forecast saved to {OUTPUT_ENRICHED}")
print(f"⏱️ Metadata enrichment completed in {time.time() - start:.2f} seconds")

# === Step 9: Send email report
python_path = sys.executable
try:
    print(f"🚀 Sending email using Python: {python_path}")
    subprocess.run([python_path, "send_email_report.py"], check=True)
except Exception as e:
    print("❌ Failed to send email report:", e)
