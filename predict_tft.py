import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import matplotlib.pyplot as plt
import os
import traceback
import warnings

warnings.filterwarnings("ignore")


print("📦 Loading data...")
df = pd.read_csv("data/processed/m5_tft_ready.csv", parse_dates=["date"])
df["price"] = df["price"].ffill().bfill().fillna(0)
df["product_id"] = df["product_id"].astype("category")
df["time_idx"] = (df["date"] - df["date"].min()).dt.days


top_products = df["product_id"].value_counts().head(10).index.tolist()
df = df[df["product_id"].isin(top_products)].copy()
print("✅ Filtered product_id categories in dataset:")
print(df["product_id"].cat.categories)


model = TemporalFusionTransformer.load_from_checkpoint("checkpoints/tft_model.ckpt")


max_encoder_length = 30
max_prediction_length = 7

training = TimeSeriesDataSet(
    df[df.time_idx < df.time_idx.max() - max_prediction_length],
    time_idx="time_idx",
    target="quantity",
    group_ids=["product_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["product_id"],
    time_varying_known_reals=["time_idx", "price", "promotion_flag"],
    time_varying_unknown_reals=["quantity"],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

print("🎯 Real-valued features used during training:", training.reals)


predict_df = df[df.time_idx >= df.time_idx.max() - (max_encoder_length + max_prediction_length)].copy()


predict_df["product_id"] = pd.Categorical(
    predict_df["product_id"],
    categories=training.get_transformer("product_id").classes_,
    ordered=True
)





try:
 prediction_dataset = TimeSeriesDataSet.from_dataset(
    training,
    predict_df,
    predict=True,
    stop_randomization=True,
   add_target_scales=True,  
)


except Exception:
    print("❌ Failed to build prediction dataset:")
    traceback.print_exc()
    exit(1)


dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
x, _ = next(iter(dataloader))

print(f"🧪 encoder_cont shape: {x['encoder_cont'].shape}")
print(f"🧪 Expected reals: {training.reals}")
print(f"🔍 Max product_id index: {x['decoder_cat'][:, :, 0].max().item()}, "
      f"Embedding size: {model.input_embeddings['product_id'].num_embeddings}")

try:
    raw_predictions, x = model.predict(dataloader, mode="raw", return_x=True)
except Exception:
    print("❌ Prediction failed:")
    traceback.print_exc()
    exit(1)

predicted_quantities = raw_predictions["prediction"].detach().cpu().numpy()
decoder_cats = x["decoder_cat"]
results = []

for i, pred in enumerate(predicted_quantities):
    product_idx = decoder_cats[i, 0, 0].item()
    product = training.get_transformer("product_id").inverse_transform(torch.tensor([product_idx]))[0]
    print(f"Product: {product} → Predicted: {pred[0].tolist()}")
    for day, value in enumerate(pred[0]):
        results.append({"product_id": product, "day": day + 1, "predicted_quantity": value})


os.makedirs("predictions", exist_ok=True)
pred_df = pd.DataFrame(results)
pred_df.to_csv("predictions/tft_forecast.csv", index=False)
print("✅ Saved predictions to predictions/tft_forecast.csv")


plt.figure(figsize=(8, 5))
for product in pred_df["product_id"].unique():
    product_data = pred_df[pred_df["product_id"] == product]
    plt.plot(product_data["day"], product_data["predicted_quantity"], marker="o", label=product)

plt.xlabel("Day")
plt.ylabel("Predicted Quantity")
plt.title("7-Day Forecast per Product")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("predictions/tft_forecast_plot.png")
plt.show()
print("✅ Forecast plot saved to predictions/tft_forecast_plot.png")
