# Some of the code is taken from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master CREDIT: Benny

import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import coacd_modified
import pyntcloud
import json
import multiprocessing
import threading
import warnings
import queue


from pathlib import Path
from tqdm import tqdm
from utils.BaseUtils import *
from utils.preprocessor import preprocess_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

import model

warnings.filterwarnings('ignore')

os.environ["PYTHONHASHSEED"] = "0"

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True


LR = 0.0001
DECAY = 1e-4
DECAY_STEP = 10
LR_DECAY = 0.7
EPOCHS = 200

LEARNING_RATE_CLIP = 1e-5
MOMENTUM_ORIGINAL = 0.1
MOMENTUM_DECCAY = 0.5
MOMENTUM_DECCAY_STEP = DECAY_STEP

BATCH_SIZE = 32


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
        


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_model.pth'))

def main():
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s.txt' % log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')


    '''MODEL LOADING'''

    num_output = 4

    shutil.copy('model/model.py', str(exp_dir))
    shutil.copy('model/pointnet2_utils.py', str(exp_dir))

    with open("plane_cache.json", "r") as plane_cache_f:
        plane_cache = json.load(plane_cache_f)

    predictor = model.get_model(num_output).cuda()
    criterion = model.get_loss().cuda()
    #predictor.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    start_epoch = 0
    #predictor = predictor.apply(weights_init)


    optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-08
    )


    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    

    print('Using', torch.cuda.get_device_name(torch.cuda.current_device()))

    best_loss = float('inf')

    for epoch in range(start_epoch, EPOCHS):

        log_string('Epoch %d (%d/%s):' %
                   (epoch + 1, epoch + 1, EPOCHS))
        '''Adjust learning rate and BN momentum'''
        lr = max(LR * (LR_DECAY **
                 (epoch // DECAY_STEP)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * \
            (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        predictor = predictor.apply(
            lambda x: bn_momentum_adjust(x, momentum))
        predictor = predictor.train()

        '''learning one epoch'''
        i = 0
        
        batch = []

        running_tloss = 0

        train_loader = load_shapenet(debug=True, data_folder="data/ShapenetRedistributed")
        validation_loader = load_shapenet(debug=False, data_folder="data/ShapenetRedistributed_val")

        processing_start = datetime.datetime.now()
        for batch in preprocess_data(train_loader, plane_cache, BATCH_SIZE):
            print("Processing time:", datetime.datetime.now()-processing_start)

            optimizer.zero_grad()
        

            points,target = batch

            if points is None:
                continue

            points = points.transpose(2, 1)

            start_time = datetime.datetime.now()
            seg_pred = predictor(
                points)
            
            #print(target)
            print(seg_pred)
            
            loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()
            print("Training time:", datetime.datetime.now()-start_time)
            running_tloss += loss.item()
            print('Training loss:', running_tloss/(i+1))
            i+=1
            print(f"{i*BATCH_SIZE} meshes processed")

            processing_start = datetime.datetime.now()


        

        #validation
        print("Running validation...")
        predictor = predictor.eval()
        running_vloss = 0
        i = 0
        with torch.no_grad():

            for batch in preprocess_data(validation_loader, plane_cache, BATCH_SIZE):

                points,target = batch

                if points is None:
                    continue

                points = points.transpose(2, 1)

                seg_pred = predictor(
                    points)
                
                loss = criterion(seg_pred,target)
                running_vloss += loss.item()
                i+=1
                print(f"{i*BATCH_SIZE} val meshes processed")
        

        print('Validation loss:', running_vloss/i)
        best = False
        if running_vloss < best_loss:
            best_loss = running_vloss
            best = True


        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, best, str(exp_dir) + '/checkpoints/', 'checkpoint.pth')


if __name__ == '__main__':
    main()
