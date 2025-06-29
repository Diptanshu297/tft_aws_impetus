import pandas as pd
import torch
import optuna
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


df = pd.read_csv("data/processed/m5_tft_ready.csv", parse_dates=["date"])
df["product_id"] = df["product_id"].astype("category")
df["time_idx"] = (df["date"] - df["date"].min()).dt.days
df["price"] = df["price"].ffill().bfill().fillna(0)


top_products = df["product_id"].value_counts().head(2).index
df = df[df["product_id"].isin(top_products)].copy()


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

validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0 )

def objective(trial):
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=trial.suggest_float("lr", 1e-4, 0.1, log=True),
        hidden_size=trial.suggest_int("hidden_size", 8, 64, step=8),
        attention_head_size=trial.suggest_int("attention_head_size", 1, 4),
        dropout=trial.suggest_float("dropout", 0.1, 0.3),
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    trainer = Trainer(
        max_epochs=20,
        gradient_clip_val=trial.suggest_float("clip_val", 0.01, 1.0),
        logger=TensorBoardLogger("lightning_logs", name=f"tft_trial_{trial.number}"),
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],
    )

    trainer.fit(model, train_loader, val_loader)
    val_loss = trainer.callback_metrics["val_loss"].item()
    return val_loss


study = optuna.create_study(direction="minimize", study_name="tft_hyperparam_tuning")
study.optimize(objective, n_trials=20, timeout=3600) 


print("🎯 Best trial:")
print(study.best_trial.params)
