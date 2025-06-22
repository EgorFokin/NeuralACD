import pandas as pd
import matplotlib.pyplot as plt

# Load the logged metrics
metrics = pd.read_csv("logs/my_model/version_5/metrics.csv")

# Filter and plot
train_loss = metrics[metrics["train_loss_epoch"].notna()]
val_loss = metrics[metrics["val_loss"].notna()]

plt.ylim(0,2)

plt.plot(train_loss["epoch"], train_loss["train_loss_epoch"], label="Train Loss")
plt.plot(val_loss["epoch"], val_loss["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.grid(True)
plt.show()