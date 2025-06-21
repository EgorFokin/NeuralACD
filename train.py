#Some of the code was taken from https://github.com/yanx27/Pointnet_Pointnet2_pytorch, credit: Xu Yan
import os
import torch
import datetime
import logging
import sys
import shutil
import numpy as np
import json
import warnings
import numpy as np
import open3d as o3d
import random
from decompose import decompose
import coacd_modified

from pathlib import Path
from utils.BaseUtils import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

#import model_cpu as model
import model.test_model as model

warnings.filterwarnings('ignore')

os.environ["PYTHONHASHSEED"] = "0"

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True

coacd_modified.set_log_level("off")
LR = 1e-03
DECAY = 1e-4
DECAY_STEP = 10
LR_DECAY = 0.7
EPOCHS = 2000

LEARNING_RATE_CLIP = 1e-5
MOMENTUM_ORIGINAL = 0.1
MOMENTUM_DECCAY = 0.5
MOMENTUM_DECCAY_STEP = DECAY_STEP

BATCH_SIZE = 64

#WEIGHTS_PATH = "log/2025-06-19_23-20/checkpoints/checkpoint.pth"

        
ROTATION = True
LIMIT = 50


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_model.pth'))

def load_point_clouds(data_folder,plane_cache,batch_size):
    batch = [[],[]]
    i = 0
    for root, _, files in os.walk(data_folder): 
        for file in files:
            if file.endswith(".npy"):
                with open(os.path.join(root, file), "rb") as f:
                    try:
                        points = np.load(f)
                    except Exception as e:
                        print(file,e)
                        continue
                    mesh_hash = file.split(".")[0]
                    if mesh_hash in plane_cache and len(plane_cache[mesh_hash]) == 5:
                        planes = plane_cache[mesh_hash]
                    else:
                        continue

                    
                    if ROTATION and random.random() < 0.75:
                        rotation = o3d.geometry.get_rotation_matrix_from_xyz(np.random.rand(3) * 2 * np.pi)
                    else:
                        rotation = np.eye(3)

                    points = np.dot(points, rotation[:3,:3].T)

                    try:
                        planes = [apply_rotation_to_plane(*plane[:4],rotation) for plane in planes]
                    except Exception as e:
                        print(file,e)
                        continue

                    if torch.isnan(torch.tensor(points)).any() or torch.isnan(torch.tensor(planes)).any():
                        continue
                    batch[0].append(points)
                    batch[1].append(planes)
                    if len(batch[0]) == batch_size:

                        points = torch.tensor(np.array(batch[0]), dtype=torch.float32).cuda()
                        target = torch.tensor(np.array(batch[1]), dtype=torch.float32).cuda()
                        yield points, target
                        i+=1
                        if i>= LIMIT:
                            return
                        batch = [[],[]]


# def get_concavity(predictor, data_folder):
#     total_concavity = 0
#     total_parts = 0
#     num_meshes = 0
#     MAX_MESHES = 10
#     for root, _, files in os.walk(data_folder): 
#         for file in files:
#             if file.endswith(".obj"):
#                 mesh = o3d.io.read_triangle_mesh(os.path.join(root, file))
#                 if len(mesh.vertices) > 10000:
#                     continue
#                 print(file)
#                 parts,_,concavity = decompose(mesh,5, predictor)
#                 total_concavity += concavity
#                 total_parts += len(parts)
#                 num_meshes += 1
#                 if num_meshes >= MAX_MESHES:
#                     return (total_concavity / num_meshes, total_parts / num_meshes)
#     return (total_concavity / num_meshes, total_parts / num_meshes)

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

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s.txt' % log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    num_output = 4

    shutil.copy('model/model.py', str(exp_dir))

    with open("plane_cache.json", "r") as plane_cache_f:
        plane_cache = json.load(plane_cache_f)

    predictor = model.get_model(num_output).cuda()
    criterion = model.get_loss().cuda()

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




    

    log_string(f'Using {torch.cuda.get_device_name(torch.cuda.current_device())}')

    best_loss = float('inf')
    loss_list = []
    val_loss_list = []
    convavity_list = []

    for epoch in range(start_epoch, EPOCHS):

        log_string('Epoch %d (%d/%s):' %
                   (epoch + 1, epoch + 1, EPOCHS))
        # lr = max(LR * (LR_DECAY **
        #          (epoch // DECAY_STEP)), LEARNING_RATE_CLIP)
        # log_string('Learning rate:%f' % lr)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        # momentum = MOMENTUM_ORIGINAL * \
        #     (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        # if momentum < 0.01:
        #     momentum = 0.01
        # log_string('BN momentum updated to: %f' % momentum)
        # predictor = predictor.apply(
        #     lambda x: bn_momentum_adjust(x, momentum))
        predictor = predictor.train()

        i = 0
        
        batch = []

        running_tloss = 0


        processing_start = datetime.datetime.now()

        for batch in load_point_clouds("data/ShapeNetPointCloud", plane_cache, BATCH_SIZE):
            log_string(f"Processing time: { datetime.datetime.now()-processing_start}")

            optimizer.zero_grad()
        

            points,target = batch

            if points is None:
                continue

            points = points.transpose(2, 1)

            start_time = datetime.datetime.now()
            seg_pred = predictor(
                points)
            
            # log_string(target)
            # log_string(seg_pred)
            
            loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()
            l =  loss.item()
            running_tloss += l
            loss_list.append(l)


            
            log_string(f"Training time: { datetime.datetime.now()-start_time}")
            log_string(f'Training loss: {running_tloss/(i+1)}')
            i+=1
            log_string(f"{i*BATCH_SIZE} meshes processed")

            processing_start = datetime.datetime.now()


        #validation
        log_string("Running validation...")
        predictor = predictor.eval()
        running_vloss = 0
        i = 0
        with torch.no_grad():

            for batch in load_point_clouds("data/ShapeNetPointCloudVal", plane_cache, BATCH_SIZE):

                points,target = batch

                if points is None:
                    continue

                points = points.transpose(2, 1)

                seg_pred = predictor(
                    points)
                
                loss = criterion(seg_pred,target)
                l = loss.item()
                running_vloss += l
                val_loss_list.append(l)
                i+=1
                log_string(f"{i*32} val meshes processed")

        

        # log_string(f'Validation loss: { running_vloss/i}')
        best = False
        if running_vloss < best_loss:
            best_loss = running_vloss
            best = True

        

        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, best, str(exp_dir) + '/checkpoints/', 'checkpoint.pth')

        # convavity = get_concavity(predictor, "data/ShapeNetMeshes")
        # convavity_list.append(convavity)

        with open(str(exp_dir) + '/history.json', 'w') as f:
            json.dump({'train': loss_list, 'val': val_loss_list, 'concavity': convavity_list}, f)


if __name__ == '__main__':
    main()
