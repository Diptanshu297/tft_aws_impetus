# ✅ train_tft.py (multi-product ready and compatible)
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os

# === Load and preprocess ===
print("\U0001F4E6 Loading data...")
df = pd.read_csv(
    "data/processed/m5_tft_ready.csv",
    parse_dates=["date"],
    usecols=["date", "product_id", "price", "promotion_flag", "quantity"],
    dtype={
        "product_id": "category",
        "price": "float32",
        "promotion_flag": "int8",
        "quantity": "float32"
    },
    low_memory=True,
    memory_map=True
)

# ✅ Optionally filter top N products
top_products = df["product_id"].value_counts().head(10).index
df = df[df["product_id"].isin(top_products)].copy()

# ✅ Preprocess
print("🛠️ Preprocessing...")
df["price"] = df["price"].ffill().bfill().fillna(0)
df["time_idx"] = (df["date"] - df["date"].min()).dt.days

# === Define encoder/predict window ===
max_encoder_length = 30
max_prediction_length = 7

# === Create training dataset
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

# ✅ Validation set
validation = TimeSeriesDataSet.from_dataset(
    training,
    df[df.time_idx >= df.time_idx.max() - max_prediction_length - max_encoder_length],
    stop_randomization=True,
)

# === Dataloaders
train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

# === TFT Model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.0005,
    hidden_size=64,
    attention_head_size=1,
    dropout=0.1,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# === Logger & Trainer
logger = TensorBoardLogger("lightning_logs", name="tft")
trainer = Trainer(
    max_epochs=20,
    gradient_clip_val=0.1,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=3),
        LearningRateMonitor("epoch")
    ],
    logger=logger,
    accelerator="auto",
)

# === Train & Save
print("🚀 Training started...")
trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
os.makedirs("checkpoints", exist_ok=True)
trainer.save_checkpoint("checkpoints/tft_model.ckpt")
print("✅ Model trained and saved at checkpoints/tft_model.ckpt")
