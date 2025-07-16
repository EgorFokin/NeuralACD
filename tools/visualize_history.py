import pandas as pd
import matplotlib.pyplot as plt

import os 
os.environ["QT_QPA_PLATFORM"] = "xcb"

versions = os.listdir("logs/lightning_logs")
versions = [int(v.split('_')[1]) for v in versions]

max_version = max(versions)

# Load the logged metrics
metrics = pd.read_csv(f"logs/lightning_logs/version_{max_version}/metrics.csv")

# Filter and plot
train_loss = metrics[metrics["train_loss"].notna()]
ema_loss = metrics[metrics["ema_loss"].notna()]
# val_loss = metrics[metrics["val_loss"].notna()]

# print("Min validation loss:",min(val_loss["val_loss"]))

# plt.ylim(0,2)

plt.plot(train_loss["train_loss"], label="Train Loss")
plt.plot(ema_loss["ema_loss"], label="EMA Loss")
# plt.plot(val_loss["epoch"], val_loss["val_loss"], label="Val Loss")
# plt.xlabel("step")
# plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss")
plt.grid(True)
plt.show()