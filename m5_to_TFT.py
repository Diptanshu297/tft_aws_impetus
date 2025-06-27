import pandas as pd
from pathlib import Path


calendar_path = Path("C:/Users/Diptanshu/OneDrive/Desktop/TFT_AWS/data/calendar.csv")
sales_path = Path("C:/Users/Diptanshu/OneDrive/Desktop/TFT_AWS/data/sales_train_validation.csv")
prices_path = Path("C:/Users/Diptanshu/OneDrive/Desktop/TFT_AWS/data/sell_prices.csv")
out_path = Path("C:/Users/Diptanshu/OneDrive/Desktop/TFT_AWS/data/processed/m5_tft_ready.csv")

def convert_m5_to_tft():
    print("🔄 Loading raw M5 data...")
    sales = pd.read_csv(sales_path)

    print("🔎 Filtering to 2 store_ids only: CA_1 and TX_1")
    sales = sales[sales["store_id"].isin(["CA_1", "TX_1"])]

    calendar = pd.read_csv(calendar_path, parse_dates=["date"])
    prices = pd.read_csv(prices_path)

    print("🔄 Converting wide to long format...")
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    value_vars = [f"d_{i}" for i in range(1, 1914)]
    sales_long = sales.melt(id_vars=id_vars, value_vars=value_vars, var_name="d", value_name="quantity")

    print("🔗 Merging with calendar and prices...")
    sales_long = sales_long.merge(calendar[["d", "date", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI"]], on="d", how="left")
    sales_long = sales_long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    print("🎯 Creating product_id and promotion flags...")
    sales_long["product_id"] = sales_long["store_id"] + "_" + sales_long["item_id"]

    
    sales_long["promotion_flag"] = 0
    sales_long.loc[sales_long["state_id"] == "CA", "promotion_flag"] = sales_long["snap_CA"]
    sales_long.loc[sales_long["state_id"] == "TX", "promotion_flag"] = sales_long["snap_TX"]
    sales_long.loc[sales_long["state_id"] == "WI", "promotion_flag"] = sales_long["snap_WI"]

    print("📦 Final formatting...")
    sales_long = sales_long.rename(columns={"sell_price": "price"})
    sales_long = sales_long[["product_id", "date", "quantity", "price", "promotion_flag"]]
    sales_long = sales_long.sort_values(["product_id", "date"])

    print(f"✅ Saving processed TFT-ready file to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sales_long.to_csv(out_path, index=False)
    print("🎉 Done!")


if __name__ == "__main__":
    convert_m5_to_tft()
