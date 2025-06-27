import json
import matplotlib.pyplot as plt
import numpy as np

TRAIN_SIZE = 10#677
VAL_SIZE = 10#76

versions = ["2025-06-25_16-52"]    

history = {"train":[], "val":[]}

for version in versions:
    with open(f"log/{version}/history.json", "r") as f:
        data = json.load(f)
        history["train"] += data["train"]
        #history["val"] += data["val"]

epochs = len(history["train"])//TRAIN_SIZE

means_train = []
means_val = []

for i in range(epochs):
    means_train.append(np.mean(history["train"][i*TRAIN_SIZE:(i+1)*TRAIN_SIZE]))
    #means_val.append(np.mean(history["val"][i*VAL_SIZE:(i+1)*VAL_SIZE]))

# means_train = means_train[60:]
# means_val = means_val[60:]

plt.ylim(top=2) 

plt.plot(means_train, label="train")
#plt.plot(means_val, label="val")

plt.legend()

plt.show()