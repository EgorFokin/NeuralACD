import json
import matplotlib.pyplot as plt
import numpy as np

TRAIN_SIZE = 46240//32
VAL_SIZE = 5152//32

# I didn't train the model in one go, so we need to combine the history of all the runs
versions = ["2025-03-10_15-39","2025-03-11_11-48", "2025-03-11_12-00", "2025-03-13_09-05","2025-03-13_15-15", "2025-03-14_11-55"]    

history = {"train":[], "val":[]}

for version in versions:
    with open(f"log/{version}/history.json", "r") as f:
        data = json.load(f)
        history["train"] += data["train"]
        history["val"] += data["val"]

epochs = len(history["train"])//TRAIN_SIZE

means_train = []
means_val = []

for i in range(epochs):
    means_train.append(np.mean(history["train"][i*TRAIN_SIZE:(i+1)*TRAIN_SIZE]))
    means_val.append(np.mean(history["val"][i*VAL_SIZE:(i+1)*VAL_SIZE]))

plt.plot(means_train, label="train")
plt.plot(means_val, label="val")

plt.legend()

plt.show()