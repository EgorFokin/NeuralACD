import sys
sys.path.append("lib/build")

from model.model import ACDModel
import h5py
import torch
from torch.utils.data import IterableDataset, DataLoader, TensorDataset
import torch.nn as nn
import json
import open3d as o3d
import random
from tqdm import tqdm
import os
from datetime import datetime
import shutil
import numpy as np
from torch.utils.data import Subset
from utils.ACDgen import ACDgen
import argparse

from utils.misc import load_config, set_seed

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import AdvancedProfiler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neural ACD model.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    print("Current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


    dataset = ACDgen(config)

    train_loader = DataLoader(dataset, batch_size=config.model.batch_size, num_workers=config.model.num_workers)


    set_seed(0)
    
    torch.set_float32_matmul_precision('high')

    model = ACDModel(learning_rate=config.model.learning_rate)

    #copy the model.py into checkpoint directory
    os.makedirs(f'checkpoints/{str(datetime.now().strftime("%d,%m,%Y-%H:%M:%S"))}/', exist_ok=True)
    shutil.copy('model/model.py', f'checkpoints/{str(datetime.now().strftime("%d,%m,%Y-%H:%M:%S"))}/model.py')

    profiler = AdvancedProfiler(dirpath="profiler_logs", filename=str(datetime.now().strftime("%d,%m,%Y-%H:%M:%S")))

    callbacks = [
        ModelCheckpoint(monitor='ema_loss',
            dirpath=f'checkpoints/{str(datetime.now().strftime("%d,%m,%Y-%H:%M:%S"))}/',
            filename='best-model-{ema_loss}',
            save_top_k=3,
            save_last=True,
            mode='min',
            every_n_train_steps=1),
            ]

    logger = CSVLogger("logs")

    trainer = pl.Trainer(
            devices=1,
            accelerator="auto",
            callbacks=callbacks,
            log_every_n_steps=10,
            logger=logger,
            max_steps=config.model.max_steps,
            #profiler=profiler,
        )

    # Start Training
    trainer.fit(
        model=model,
        train_dataloaders=train_loader
    )
