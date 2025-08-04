import pandas as pd
import matplotlib.pyplot as plt
import os
import time

os.environ["QT_QPA_PLATFORM"] = "xcb"
plt.ion()  # Turn on interactive mode

fig, ax = plt.subplots()
line1, = ax.plot([], [], label="Train Loss")
line2, = ax.plot([], [], label="EMA Loss")
ax.set_title("Training Loss")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend()

while True:
    versions = os.listdir("logs/lightning_logs")
    versions = [int(v.split('_')[1]) for v in versions if v.startswith("version_")]
    if not versions:
        continue
    max_version = max(versions)

    try:
        metrics = pd.read_csv(f"logs/lightning_logs/version_{max_version}/metrics.csv")
    except Exception as e:
        time.sleep(1)
        continue  # File might be locked or incomplete

    train_loss = metrics[metrics["train_loss"].notna()]
    ema_loss = metrics[metrics["ema_loss"].notna()]

    if len(train_loss) == 0:
        continue

    line1.set_xdata(train_loss["step"])
    line1.set_ydata(train_loss["train_loss"])
    line2.set_xdata(ema_loss["step"])
    line2.set_ydata(ema_loss["ema_loss"])

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

    time.sleep(1)  # Refresh every second
