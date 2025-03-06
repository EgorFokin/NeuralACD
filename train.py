import os
import torch
import datetime
import logging
import sys
import shutil
import numpy as np
import json
import warnings

import open3d as o3d


from pathlib import Path
from utils.BaseUtils import *
from utils.preprocessor import preprocess_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

#import model_cpu as model
import model

warnings.filterwarnings('ignore')

os.environ["PYTHONHASHSEED"] = "0"

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

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

WEIGHTS_PATH = "C:\\Users\\egorf\\Desktop\\cmpt469\\DeepConvexDecomposition\\log\\2025-03-01_17-04\\checkpoints\\best_model.pth"




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


    '''MODEL LOADING'''

    num_output = 4

    shutil.copy('model/model.py', str(exp_dir))
    shutil.copy('model/pointnet2_utils.py', str(exp_dir))

    with open("plane_cache.json", "r") as plane_cache_f:
        plane_cache = json.load(plane_cache_f)

    predictor = model.get_model(num_output).cuda()
    criterion = model.get_loss().cuda()
    #predictor.apply(inplace_relu)


    start_epoch = 0

    optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-08
    )

    
    try:
        checkpoint = torch.load(WEIGHTS_PATH)
        predictor.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        log_string("Loaded model from checkpoint")
    except Exception as e:
        log_string(f"Failed to load model from checkpoint: {e}")



    

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    

    log_string(f'Using{torch.cuda.get_device_name(torch.cuda.current_device())}')

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
        log_string('BN momentum updated to: %f' % momentum)
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
            log_string(f"Processing time: { datetime.datetime.now()-processing_start}")

            optimizer.zero_grad()
        

            points,target = batch

            if points is None:
                continue

            points = points.transpose(2, 1)

            start_time = datetime.datetime.now()
            seg_pred = predictor(
                points)
            
            #log_string(target)
            #log_string(seg_pred)
            
            loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()
            log_string(f"Training time: { datetime.datetime.now()-start_time}")
            running_tloss += loss.item()
            log_string(f'Training loss: {running_tloss/(i+1)}')
            i+=1
            log_string(f"{i*BATCH_SIZE} meshes processed")

            del batch, points, target, seg_pred, loss
            torch.cuda.empty_cache()

            processing_start = datetime.datetime.now()
          
            

        

        #validation
        log_string("Running validation...")
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
                log_string(f"{i*BATCH_SIZE} val meshes processed")
        

        log_string(f'Validation loss: { running_vloss/i}')
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
